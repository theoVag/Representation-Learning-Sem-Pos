
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
from tqdm import tqdm
import sys

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from utilities_custom_loc import create_loader_classification

def save_model(epochs, model, optimizer, criterion,folder_name='models_brain'):

    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(folder_name,"best_model_checkpoint.pth"))
    

def valid_class_pos(validation_loader,model,device,criterion):
    
    val_loss=0
    step=0

    f1_batch=0
    model.eval()
            
    with torch.no_grad():
        for i, (images,y,loc) in enumerate(validation_loader):
            images = torch.cat(images, 0)
            y = torch.cat(y, 0)
            loc = torch.cat(loc, 0)
            step+=1
            images, y, loc = images.to(device), y.to(device), loc.to(device)
            y=y.float()
            outputs = model(images,loc)
            y = nn.functional.one_hot(y.long(),num_classes=2)
            y=torch.squeeze(y)
            y = y.float().to(device)
            outputs=outputs.to(device)
            loss = criterion(outputs,y).item()#/batch_dim
            y = torch.argmax(y,dim=1)
            outputs = torch.argmax(outputs,dim=1)            
            y_num=y.cpu()
            pred = outputs.cpu()
            f1_batch += f1_score(y_num, pred,zero_division=1)
            val_loss+=loss
    val_loss = val_loss/len(validation_loader)
    val_f1 = f1_batch/len(validation_loader)
    return val_loss,val_f1


def trainer_classification_pos(model,num_epochs, criterion,optimizer,device,models_dir,init_epoch=0,data_info_train=None,data_info_val=None,wandb=None):

    res_training_loss = []
    res_validation_loss = []

    accuracy_batch=0
    precision_batch=0
    f1_batch=0
    recall_batch=0
    best_val_metric=0
    no_improvement_count=0
    patience=200
    for epoch in tqdm(range(init_epoch+1, num_epochs)): # bale tqdm na blepeis
        training_loss=0
        batch_loss=0
        accuracy_batch=0
        precision_batch=0
        f1_batch=0
        recall_batch=0
        model.train()
        if epoch%data_info_train['epoch_step']==0 or epoch==1:
            if epoch!=1:
                del train_dataset
                del train_loader
                del validation_dataset
                del validation_loader
                gc.collect()

            train_dataset,train_loader = create_loader_classification(data_info_train)
            validation_dataset,validation_loader = create_loader_classification(data_info_val)
        
        for i, (images,y,loc) in enumerate(train_loader):

            images = torch.cat(images, 0)
            y = torch.cat(y, 0)
            loc = torch.cat(loc, 0)
            images, y, loc = images.to(device), y.to(device), loc.to(device)
            #y = y.float()
            outputs = model(images,loc)
            y = nn.functional.one_hot(y.long(),num_classes=2)
            y=torch.squeeze(y)
            y = y.float().to(device)
            outputs=outputs.to(device)
            loss = criterion(outputs,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            y = torch.argmax(y,dim=1)
            outputs = torch.argmax(outputs,dim=1)
            pred = outputs.cpu().detach().numpy()
            batch_loss += loss.item()
            y_num=y.cpu()
            accuracy_batch += accuracy_score(y_num, pred)
            precision_batch += precision_score(y_num, pred)
            recall_batch += recall_score(y_num, pred)
            f1_batch += f1_score(y_num, pred)
        
        training_loss = batch_loss/len(train_loader)
        training_acc = accuracy_batch/len(train_loader)
        training_precision = precision_batch/len(train_loader)
        training_recall = recall_batch/len(train_loader)
        training_f1 = f1_batch/len(train_loader)

        wandb.log({"training_acc": training_acc, "epoch": epoch})
        wandb.log({"train_loss": training_loss, "epoch": epoch})
        wandb.log({"training prec": training_precision, "epoch": epoch})
        wandb.log({"training recall": training_recall, "epoch": epoch})
        wandb.log({"training f1": training_f1, "epoch": epoch})

        # validation step 
        validation_loss = 0        
        validation_loss,val_f1 = valid_class_pos(validation_loader,model,device,criterion)
        wandb.log({"validation_loss": validation_loss, "epoch": epoch})
        wandb.log({"validation f1": val_f1, "epoch": epoch})
        print(f'epoch: {epoch+1} / {num_epochs}, loss: {training_loss:.4f}, val_loss: {validation_loss:.4f}')
        res_training_loss.append(training_loss)
        res_validation_loss.append(validation_loss)

        if epoch==init_epoch+1:
            best_validation_loss = validation_loss
            save_model(epoch,model,optimizer,criterion,models_dir)
        
        if val_f1>best_val_metric:
            best_validation_loss = validation_loss 
            best_val_metric = val_f1
            save_model(epoch,model,optimizer,criterion,models_dir)
            no_improvement_count=0
        else:
            no_improvement_count += 1
        
        if no_improvement_count>patience:
            break

    print("Finished Training")
    return best_val_metric

