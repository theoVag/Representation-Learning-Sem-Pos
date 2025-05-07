#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 14:17:23 2024

@author: tpv
"""

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gc
from tqdm import tqdm

from torch.optim import lr_scheduler
from utilities_custom_loc import create_loader  # ,create_loader_supervised
from lightly.utils.scheduler import cosine_schedule
from lightly.loss.vicreg_loss import invariance_loss, variance_loss, covariance_loss


def save_model(epochs, model, optimizer, folder_name="models_brain"):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        folder_name + "/best_model_checkpoint.pth",
    )  # folder_name+'/model_'+str(epochs)+'.pth')


def trainer_vicreg_loc( model,
                        num_epochs,
                        optimizer,
                        scheduler,
                        device,
                        models_dir,
                        init_epoch=0,
                        data_info_train=None,
                        data_info_val=None,
                        wandb=None,
                        ):

    res_training_loss = []
    res_validation_loss = []
    no_improvement = 0
    patience = 800
    conv1_grad_norm = 0
    conv2_grad_norm = 0
    for epoch in tqdm(range(init_epoch + 1, num_epochs)):  # bale tqdm na blepeis
        training_loss = 0
        batch_loss = 0
        inv_loss_full = 0
        var_loss_full = 0
        cov_loss_full = 0
        step = 0
        model.train()
        if epoch % data_info_train["epoch_step"] == 0 or epoch == init_epoch + 1:
            if epoch != init_epoch + 1:
                del train_dataset
                del train_loader
                del validation_dataset
                del validation_loader
                gc.collect()

            #data_info_train["cur_epoch"] = epoch % data_info_train["epoch_step"]  # 800
            #data_info_val["cur_epoch"] = epoch % data_info_train["epoch_step"]
            train_dataset, train_loader = create_loader(data_info_train)
            validation_dataset, validation_loader = create_loader(data_info_val)

        for i, batch in enumerate(train_loader):
            step += 1
            views_temp, loc_views = batch  # [0]
            x0, x1 = views_temp
            x2, x3 = loc_views

            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            z0 = model(x0, x2)
            z1 = model(x1, x3)

            # VICReg loss
            inv_loss = invariance_loss(x=z0, y=z1)
            var_loss = 0.5 * (variance_loss(x=z0) + variance_loss(x=z1))
            cov_loss = covariance_loss(x=z0) + covariance_loss(x=z1)
            loss = 25.0 * inv_loss + 25.0 * var_loss + 1.0 * cov_loss

            inv_loss_full += inv_loss.detach().item()
            var_loss_full += var_loss.detach().item()
            cov_loss_full += cov_loss.detach().item()
            batch_loss += loss.detach().item()
            loss.backward()

            # Log gradients of the first and second convolutional layers
            conv1_grad_norm = model.backbone.layer1[0].conv2.weight.grad.norm().item()
            conv2_grad_norm = model.backbone.layer4[0].conv2.weight.grad.norm().item()
            wandb.log(
                {"conv1_grad_norm": conv1_grad_norm, "conv4_grad_norm": conv2_grad_norm}
            )

            max_grad_norm = 10
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            if scheduler != None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"current_lr": current_lr})
            optimizer.zero_grad()

        training_loss = batch_loss / len(train_loader)

        wandb.log({"train epoch_loss": training_loss, "epoch": epoch})
        wandb.log({"train inv_loss": inv_loss_full, "epoch": epoch})
        wandb.log({"train var_loss": var_loss_full, "epoch": epoch})
        wandb.log({"train cov_loss": cov_loss_full, "epoch": epoch})

        # validation step
        validation_loss = 0
        val_steps = 0
        model.eval()
        batch_loss = 0
        for i, batch in enumerate(validation_loader):
            with torch.no_grad():
                views_temp, loc_views = batch  # [0]
                x0, x1 = views_temp
                x2, x3 = loc_views

                x0 = x0.to(device)
                x1 = x1.to(device)
                x2 = x2.to(device)
                x3 = x3.to(device)

                z0 = model(x0, x2)
                z1 = model(x1, x3)
                # loss = criterion(z0, z1)
                inv_loss = invariance_loss(x=z0, y=z1)
                var_loss = 0.5 * (variance_loss(x=z0) + variance_loss(x=z1))
                cov_loss = covariance_loss(x=z0) + covariance_loss(x=z1)
                loss = 25.0 * inv_loss + 25.0 * var_loss + 1.0 * cov_loss
                batch_loss += loss.detach().item()

        validation_loss = batch_loss / len(validation_loader)
        wandb.log({"val epoch_loss": validation_loss, "epoch": epoch})
        print(
            f"epoch: {epoch+1} / {num_epochs}, loss: {training_loss:.4f}, val_loss: {validation_loss:.4f}"
        )

        res_training_loss.append(training_loss)
        res_validation_loss.append(validation_loss)

        if epoch == data_info_train["epoch"] + 1:
            best_validation_loss = validation_loss
            if epoch == 1:
                save_model(epoch, model, optimizer, models_dir)

        if (
            validation_loss < best_validation_loss
        ):  # edw to bazw megalutero alla thelei allagh
            best_validation_loss = validation_loss
            # best_validation_metric = validation_metric
            save_model(epoch, model, optimizer, models_dir)
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement > patience:
            break

        # scheduler.step(validation_loss)
    print("Finished Training")









# DELETE BELOW

def trainer_swav_loc(
    model,
    num_epochs,
    batch_size,
    spatial_size,
    num_sub_vol,
    learning_rate,
    criterion,
    optimizer,
    scheduler,
    device,
    models_dir,
    writer,
    init_epoch=0,
    data_info_train=None,
    data_info_val=None,
    wandb=None,
):

    ###scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, cooldown=2)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_total_steps, eta_min=10^(-7))

    res_training_loss = []
    res_validation_loss = []

    for epoch in tqdm(range(init_epoch + 1, num_epochs)):  # bale tqdm na blepeis
        training_loss = 0
        batch_loss = 0
        step = 0
        model.train()
        if epoch % data_info_train["epoch_step"] == 0 or epoch == init_epoch + 1:
            if epoch != init_epoch + 1:
                del train_dataset
                del train_loader
                del validation_dataset
                del validation_loader
                gc.collect()

            """train_dataset=None
            train_loader=None
            validation_dataset=None
            validation_loader=None"""
            print("load data...")
            data_info_train["cur_epoch"] = epoch
            data_info_val["cur_epoch"] = epoch
            train_dataset, train_loader = create_loader(data_info_train)

            validation_dataset, validation_loader = create_loader(data_info_val)

        for i, batch in enumerate(train_loader):
            step += 1
            views, loc = batch  # [0]
            model.prototypes.normalize()
            multi_crop_features = [
                model(view.to(device), loc.to(device)) for view, loc in zip(views, loc)
            ]
            high_resolution = multi_crop_features[:2]
            low_resolution = multi_crop_features[2:]
            loss = criterion(high_resolution, low_resolution)

            batch_loss += loss.detach().item()
            loss.backward()
            # Log gradients of the first and second convolutional layers
            conv1_grad_norm = model.backbone.layer1[0].conv2.weight.grad.norm().item()
            conv2_grad_norm = model.backbone.layer4[0].conv2.weight.grad.norm().item()
            wandb.log(
                {"conv1_grad_norm": conv1_grad_norm, "conv4_grad_norm": conv2_grad_norm}
            )
            optimizer.step()
            optimizer.zero_grad()
            max_grad_norm = 10

            # kleinw kai auto na doume
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            # to kalo einai

            # kalo autoooptimizer.zero_grad()

            # allagh gia scheduler edw
            if scheduler != None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"current_lr": current_lr})
            optimizer.zero_grad()

        training_loss = batch_loss / len(train_loader)

        # training_loss = training_loss/step

        # writer.add_scalar('training loss', training_loss, epoch)

        wandb.log({"train epoch_loss": training_loss, "epoch": epoch})

        # validation step
        validation_loss = 0
        val_steps = 0

        # validation_loss = valid(validation_loader,model,device,criterion)
        # add--------
        model.eval()
        # CosineEmbeddingLoss

        batch_loss = 0
        for i, batch in enumerate(validation_loader):
            with torch.no_grad():
                views, loc = batch  # [0]
                model.prototypes.normalize()
                multi_crop_features = [
                    model(view.to(device), loc.to(device))
                    for view, loc in zip(views, loc)
                ]
                high_resolution = multi_crop_features[:2]
                low_resolution = multi_crop_features[2:]
                loss = criterion(high_resolution, low_resolution)

                batch_loss += loss.detach().item()
                # loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()

        validation_loss = batch_loss / len(validation_loader)
        # print(batch_loss)

        # here ----
        # val_steps = len(validation_loader)
        # validation_loss = validation_loss/(val_steps)

        # writer.add_scalar('validation loss', validation_loss , epoch)
        wandb.log({"val epoch_loss": validation_loss, "epoch": epoch})

        print(
            f"epoch: {epoch+1} / {num_epochs}, loss: {training_loss:.4f}, val_loss: {validation_loss:.4f}"
        )

        res_training_loss.append(training_loss)
        res_validation_loss.append(validation_loss)
        # res_validation_metric.append(validation_metric)
        # res_validation_dice.append(validation_dice)

        if epoch == init_epoch + 1:
            best_validation_loss = validation_loss  # /len(validation_loader)
            # best_validation_metric = validation_metric
            if epoch == 1:
                save_model(epoch, model, optimizer, criterion, models_dir)

        if (
            validation_loss < best_validation_loss
        ):  # edw to bazw megalutero alla thelei allagh
            best_validation_loss = validation_loss
            # best_validation_metric = validation_metric
            save_model(epoch, model, optimizer, criterion, models_dir)

        # scheduler.step(validation_loss)

    print("Finished Training")


def trainer_simclr_loc(
    model,
    num_epochs,
    batch_size,
    spatial_size,
    num_sub_vol,
    learning_rate,
    criterion,
    optimizer,
    scheduler,
    device,
    models_dir,
    writer,
    init_epoch=0,
    data_info_train=None,
    data_info_val=None,
    wandb=None,
):

    ###scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, cooldown=2)

    print("device inside: ", device)
    # n_total_steps=len(train_loader) * num_sub_vol
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_total_steps, eta_min=10^(-7))

    res_training_loss = []
    res_validation_loss = []
    no_improvement = 0
    patience = 800
    conv1_grad_norm = 0
    conv2_grad_norm = 0
    for epoch in tqdm(range(init_epoch + 1, num_epochs)):  # bale tqdm na blepeis
        print(epoch)
        training_loss = 0
        batch_loss = 0
        step = 0
        model.train()
        if epoch % data_info_train["epoch_step"] == 0 or epoch == init_epoch + 1:
            if epoch != init_epoch + 1:
                del train_dataset
                del train_loader
                del validation_dataset
                del validation_loader
                gc.collect()

            data_info_train["cur_epoch"] = epoch % 3800
            data_info_val["cur_epoch"] = epoch % 3800
            train_dataset, train_loader = create_loader(data_info_train)

            validation_dataset, validation_loader = create_loader(data_info_val)

        for i, batch in enumerate(train_loader):
            step += 1
            views_temp, loc_views = batch  # [0]
            x0, x1 = views_temp
            x2, x3 = loc_views

            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)

            z0 = model(x0, x2)
            z1 = model(x1, x3)

            # z0 = nn.functional.normalize(z0, p=2, dim=1)
            # z1 = nn.functional.normalize(z1, p=2, dim=1)

            loss = criterion(z0, z1)  #

            batch_loss += loss.detach().item()
            loss.backward()
            # Log gradients of the first and second convolutional layers
            conv1_grad_norm = model.backbone.layer1[0].conv2.weight.grad.norm().item()
            conv2_grad_norm = model.backbone.layer4[0].conv2.weight.grad.norm().item()
            wandb.log(
                {"conv1_grad_norm": conv1_grad_norm, "conv4_grad_norm": conv2_grad_norm}
            )

            max_grad_norm = 10

            # kleinw kai auto na doume
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            if scheduler != None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"current_lr": current_lr})
            optimizer.zero_grad()

        training_loss = batch_loss / len(train_loader)

        # writer.add_scalar('training loss', training_loss, epoch)
        # wandb.log({"train epoch_loss": training_loss})
        wandb.log({"train epoch_loss": training_loss, "epoch": epoch})
        # validation step
        validation_loss = 0
        model.eval()
        batch_loss = 0
        for i, batch in enumerate(validation_loader):
            with torch.no_grad():
                views_temp, loc_views = batch  # [0]
                x0, x1 = views_temp
                x2, x3 = loc_views

                x0 = x0.to(device)
                x1 = x1.to(device)
                x2 = x2.to(device)
                x3 = x3.to(device)

                z0 = model(x0, x2)
                z1 = model(x1, x3)
                loss = criterion(z0, z1)

                batch_loss += loss.detach().item()

        validation_loss = batch_loss / len(validation_loader)

        wandb.log({"val epoch_loss": validation_loss, "epoch": epoch})

        writer.add_scalar("validation loss", validation_loss, epoch)
        print(
            f"epoch: {epoch+1} / {num_epochs}, loss: {training_loss:.4f}, val_loss: {validation_loss:.4f}"
        )

        res_training_loss.append(training_loss)
        res_validation_loss.append(validation_loss)
        # res_validation_metric.append(validation_metric)
        # res_validation_dice.append(validation_dice)

        if epoch == data_info_train["epoch"] + 1:
            best_validation_loss = validation_loss  # /len(validation_loader)
            # best_validation_metric = validation_metric
            if epoch == 1:
                save_model(epoch, model, optimizer, criterion, models_dir)

        if (
            validation_loss < best_validation_loss
        ):  # edw to bazw megalutero alla thelei allagh
            best_validation_loss = validation_loss
            # best_validation_metric = validation_metric
            save_model(epoch, model, optimizer, criterion, models_dir)
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement > patience:
            break

        # scheduler.step(validation_loss)

    print("Finished Training")


def trainer_simclr_sup_loc(
    model,
    num_epochs,
    batch_size,
    spatial_size,
    num_sub_vol,
    learning_rate,
    criterion,
    optimizer,
    scheduler,
    device,
    models_dir,
    writer,
    init_epoch=0,
    data_info_train=None,
    data_info_val=None,
    wandb=None,
):

    ###scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, cooldown=2)

    print("device inside: ", device)
    # n_total_steps=len(train_loader) * num_sub_vol
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_total_steps, eta_min=10^(-7))

    res_training_loss = []
    res_validation_loss = []
    no_improvement = 0
    patience = 800
    conv1_grad_norm = 0
    conv2_grad_norm = 0
    for epoch in tqdm(range(init_epoch + 1, num_epochs)):  # bale tqdm na blepeis
        print(epoch)
        training_loss = 0
        batch_loss = 0
        step = 0
        model.train()
        if epoch % data_info_train["epoch_step"] == 0 or epoch == init_epoch + 1:
            if epoch != init_epoch + 1:
                del train_dataset
                del train_loader
                del validation_dataset
                del validation_loader
                gc.collect()

            data_info_train["cur_epoch"] = epoch % 3800
            data_info_val["cur_epoch"] = epoch % 3800
            train_dataset, train_loader = create_loader_supervised(data_info_train)

            validation_dataset, validation_loader = create_loader_supervised(
                data_info_val
            )

        for i, batch in enumerate(train_loader):
            step += 1
            views_temp = batch  # [0]
            x0, y0, loc0 = views_temp

            x0 = x0.to(device)
            y0 = y0.to(device)
            loc0 = loc0.to(device)

            z0 = model(x0, loc0)
            # z1 = model(x1)

            # z0 = nn.functional.normalize(z0, p=2, dim=1)
            # z1 = nn.functional.normalize(z1, p=2, dim=1)

            loss = criterion(z0, y0)  #

            batch_loss += loss.detach().item()
            loss.backward()
            # Log gradients of the first and second convolutional layers
            conv1_grad_norm = model.backbone.layer1[0].conv2.weight.grad.norm().item()
            conv2_grad_norm = model.backbone.layer4[0].conv2.weight.grad.norm().item()
            wandb.log(
                {"conv1_grad_norm": conv1_grad_norm, "conv4_grad_norm": conv2_grad_norm}
            )

            max_grad_norm = 10

            # kleinw kai auto na doume
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            if scheduler != None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"current_lr": current_lr})
            optimizer.zero_grad()

        training_loss = batch_loss / len(train_loader)

        # writer.add_scalar('training loss', training_loss, epoch)
        # wandb.log({"train epoch_loss": training_loss})
        wandb.log({"train epoch_loss": training_loss, "epoch": epoch})
        # validation step
        validation_loss = 0
        model.eval()
        batch_loss = 0
        for i, batch in enumerate(validation_loader):
            with torch.no_grad():
                views_temp = batch  # [0]
                x0, y0, loc0 = views_temp

                x0 = x0.to(device)
                y0 = y0.to(device)
                loc0 = loc0.to(device)

                z0 = model(x0, loc0)
                loss = criterion(z0, y0)

                batch_loss += loss.detach().item()

        validation_loss = batch_loss / len(validation_loader)

        wandb.log({"val epoch_loss": validation_loss, "epoch": epoch})

        writer.add_scalar("validation loss", validation_loss, epoch)
        print(
            f"epoch: {epoch+1} / {num_epochs}, loss: {training_loss:.4f}, val_loss: {validation_loss:.4f}"
        )

        res_training_loss.append(training_loss)
        res_validation_loss.append(validation_loss)
        # res_validation_metric.append(validation_metric)
        # res_validation_dice.append(validation_dice)

        if epoch == data_info_train["epoch"] + 1:
            best_validation_loss = validation_loss  # /len(validation_loader)
            # best_validation_metric = validation_metric
            if epoch == 1:
                save_model(epoch, model, optimizer, criterion, models_dir)

        if (
            validation_loss < best_validation_loss
        ):  # edw to bazw megalutero alla thelei allagh
            best_validation_loss = validation_loss
            # best_validation_metric = validation_metric
            save_model(epoch, model, optimizer, criterion, models_dir)
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement > patience:
            break

        # scheduler.step(validation_loss)

    print("Finished Training")


from lightly.loss.memory_bank import MemoryBankModule
from lightly.models import utils


def trainer_smog_loc(
    model,
    num_epochs,
    batch_size,
    spatial_size,
    num_sub_vol,
    learning_rate,
    criterion,
    optimizer,
    scheduler,
    device,
    models_dir,
    writer,
    init_epoch=0,
    data_info_train=None,
    data_info_val=None,
    wandb=None,
):

    ###scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, cooldown=2)
    memory_bank_size = 300 * batch_size
    memory_bank = MemoryBankModule(size=memory_bank_size)
    print("device inside: ", device)
    # n_total_steps=len(train_loader) * num_sub_vol
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_total_steps, eta_min=10^(-7))

    res_training_loss = []
    res_validation_loss = []
    no_improvement = 0
    patience = 800
    conv1_grad_norm = 0
    conv2_grad_norm = 0
    for epoch in tqdm(range(init_epoch + 1, num_epochs)):  # bale tqdm na blepeis
        print(epoch)
        training_loss = 0
        batch_loss = 0
        step = 0
        model.train()
        if epoch % data_info_train["epoch_step"] == 0 or epoch == init_epoch + 1:
            if epoch != init_epoch + 1:
                del train_dataset
                del train_loader
                del validation_dataset
                del validation_loader
                gc.collect()

            data_info_train["cur_epoch"] = epoch % 3800
            data_info_val["cur_epoch"] = epoch % 3800
            train_dataset, train_loader = create_loader(data_info_train)

            validation_dataset, validation_loader = create_loader(data_info_val)

        for i, batch in enumerate(train_loader):
            step += 1
            views_temp, loc_views = batch  # [0]
            x0, x1 = views_temp
            x2, x3 = loc_views

            if i % 2:
                # swap batches every second iteration
                x1, x0 = x0, x1
                x3, x2 = x2, x3

            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)

            if step > 0 and step % 300 == 0:
                # reset group features and weights every 300 iterations
                model.reset_group_features(memory_bank=memory_bank)
                model.reset_momentum_weights()
            else:
                # update momentum
                utils.update_momentum(model.backbone, model.backbone_momentum, 0.99)
                utils.update_momentum(
                    model.projection_head, model.projection_head_momentum, 0.99
                )

            x0_encoded, x0_predicted = model(x0, x2)
            x1_encoded = model.forward_momentum(x1, x3)

            # update group features and get group assignments
            assignments = model.smog.assign_groups(x1_encoded)
            group_features = model.smog.get_updated_group_features(x0_encoded)
            logits = model.smog(x0_predicted, group_features, temperature=0.1)
            model.smog.set_group_features(group_features)

            loss = criterion(logits, assignments)

            # use memory bank to periodically reset the group features with k-means
            memory_bank(x0_encoded, update=True)

            batch_loss += loss.detach().item()
            loss.backward()
            # Log gradients of the first and second convolutional layers
            conv1_grad_norm = model.backbone.layer1[0].conv2.weight.grad.norm().item()
            conv2_grad_norm = model.backbone.layer4[0].conv2.weight.grad.norm().item()
            wandb.log(
                {"conv1_grad_norm": conv1_grad_norm, "conv4_grad_norm": conv2_grad_norm}
            )

            max_grad_norm = 10

            # kleinw kai auto na doume
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            if scheduler != None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"current_lr": current_lr})
            optimizer.zero_grad()

        training_loss = batch_loss / len(train_loader)

        # writer.add_scalar('training loss', training_loss, epoch)
        # wandb.log({"train epoch_loss": training_loss})
        wandb.log({"train epoch_loss": training_loss, "epoch": epoch})
        # validation step
        validation_loss = 0
        model.eval()
        batch_loss = 0
        for i, batch in enumerate(validation_loader):
            with torch.no_grad():
                views_temp, loc_views = batch  # [0]
                x0, x1 = views_temp
                x2, x3 = loc_views

                if i % 2:
                    # swap batches every second iteration
                    x1, x0 = x0, x1
                    x3, x2 = x2, x3

                x0 = x0.to(device)
                x1 = x1.to(device)
                x2 = x2.to(device)
                x3 = x3.to(device)

                if step > 0 and step % 300 == 0:
                    # reset group features and weights every 300 iterations
                    model.reset_group_features(memory_bank=memory_bank)
                    model.reset_momentum_weights()
                else:
                    # update momentum
                    utils.update_momentum(model.backbone, model.backbone_momentum, 0.99)
                    utils.update_momentum(
                        model.projection_head, model.projection_head_momentum, 0.99
                    )

                x0_encoded, x0_predicted = model(x0, x2)
                x1_encoded = model.forward_momentum(x1, x3)

                # update group features and get group assignments
                assignments = model.smog.assign_groups(x1_encoded)
                group_features = model.smog.get_updated_group_features(x0_encoded)
                logits = model.smog(x0_predicted, group_features, temperature=0.1)
                model.smog.set_group_features(group_features)

                loss = criterion(logits, assignments)

                # use memory bank to periodically reset the group features with k-means
                memory_bank(x0_encoded, update=True)

                batch_loss += loss.detach().item()

        validation_loss = batch_loss / len(validation_loader)

        wandb.log({"val epoch_loss": validation_loss, "epoch": epoch})

        writer.add_scalar("validation loss", validation_loss, epoch)
        print(
            f"epoch: {epoch+1} / {num_epochs}, loss: {training_loss:.4f}, val_loss: {validation_loss:.4f}"
        )

        res_training_loss.append(training_loss)
        res_validation_loss.append(validation_loss)
        # res_validation_metric.append(validation_metric)
        # res_validation_dice.append(validation_dice)

        if epoch == data_info_train["epoch"] + 1:
            best_validation_loss = validation_loss  # /len(validation_loader)
            # best_validation_metric = validation_metric
            if epoch == 1:
                save_model(epoch, model, optimizer, criterion, models_dir)

        if (
            validation_loss < best_validation_loss
        ):  # edw to bazw megalutero alla thelei allagh
            best_validation_loss = validation_loss
            # best_validation_metric = validation_metric
            save_model(epoch, model, optimizer, criterion, models_dir)
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement > patience:
            break

        # scheduler.step(validation_loss)

    print("Finished Training")


def trainer_dino_loc(
    model,
    num_epochs,
    batch_size,
    spatial_size,
    num_sub_vol,
    learning_rate,
    criterion,
    optimizer,
    scheduler,
    device,
    models_dir,
    writer,
    init_epoch=0,
    data_info_train=None,
    data_info_val=None,
    wandb=None,
):

    ###scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, cooldown=2)

    print("device inside: ", device)
    # n_total_steps=len(train_loader) * num_sub_vol
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_total_steps, eta_min=10^(-7))

    res_training_loss = []
    res_validation_loss = []
    no_improvement = 0
    patience = 800
    conv1_grad_norm = 0
    conv2_grad_norm = 0
    for epoch in tqdm(range(init_epoch + 1, num_epochs)):  # bale tqdm na blepeis
        print(epoch)
        training_loss = 0
        batch_loss = 0
        step = 0
        model.train()
        if epoch % data_info_train["epoch_step"] == 0 or epoch == init_epoch + 1:
            if epoch != init_epoch + 1:
                del train_dataset
                del train_loader
                del validation_dataset
                del validation_loader
                gc.collect()

            data_info_train["cur_epoch"] = epoch % 3800
            data_info_val["cur_epoch"] = epoch % 3800
            train_dataset, train_loader = create_loader(data_info_train)

            validation_dataset, validation_loader = create_loader(data_info_val)
        momentum_val = cosine_schedule(epoch, num_epochs, 0.996, 1)
        for i, batch in enumerate(train_loader):
            step += 1
            # views_temp,loc_views = batch#[0]
            # x0, x1=views_temp
            # x2, x3=loc_views
            views_temp, loc_temp = batch  # [0]
            # model.prototypes.normalize()
            # multi_crop_features = [model(view.to(device),loc.to(device)) for view,loc in zip(views,loc)]

            update_momentum(
                model.student_backbone, model.teacher_backbone, m=momentum_val
            )
            update_momentum(model.student_head, model.teacher_head, m=momentum_val)

            global_views = views_temp[:2]
            global_loc_views = loc_temp[:2]
            teacher_out = [
                model.forward_teacher(view.to(device), loc.to(device))
                for view, loc in zip(global_views, global_loc_views)
            ]
            student_out = [
                model.forward(view.to(device), loc.to(device))
                for view, loc in zip(views_temp, loc_temp)
            ]
            # print(type(teacher_out))
            # print(teacher_out.shape)
            # check device of tenssors
            print("device")
            print("teacher", teacher_out[0].device)
            print("student", student_out[0].device)

            loss = criterion(teacher_out, student_out, epoch=epoch)

            batch_loss += loss.detach().item()
            loss.backward()
            model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
            # Log gradients of the first and second convolutional layers
            conv1_grad_norm = (
                model.student_backbone.layer1[0].conv2.weight.grad.norm().item()
            )
            conv2_grad_norm = (
                model.student_backbone.layer4[0].conv2.weight.grad.norm().item()
            )
            wandb.log(
                {"conv1_grad_norm": conv1_grad_norm, "conv4_grad_norm": conv2_grad_norm}
            )

            max_grad_norm = 10

            # kleinw kai auto na doume
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            if scheduler != None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"current_lr": current_lr})
            optimizer.zero_grad()

        training_loss = batch_loss / len(train_loader)

        # writer.add_scalar('training loss', training_loss, epoch)
        # wandb.log({"train epoch_loss": training_loss})
        wandb.log({"train epoch_loss": training_loss, "epoch": epoch})
        # validation step
        validation_loss = 0
        model.eval()
        batch_loss = 0
        for i, batch in enumerate(validation_loader):
            with torch.no_grad():
                views, loc = batch  # [0]
                # model.prototypes.normalize()
                # multi_crop_features = [model(view.to(device),loc.to(device)) for view,loc in zip(views,loc)]

                update_momentum(
                    model.student_backbone, model.teacher_backbone, m=momentum_val
                )
                update_momentum(model.student_head, model.teacher_head, m=momentum_val)

                global_views = views[:2]
                global_loc_views = loc[:2]
                teacher_out = [
                    model.forward_teacher(view.to(device), loc.to(device))
                    for view, loc in zip(global_views, global_loc_views)
                ]
                student_out = [
                    model.forward(view.to(device), loc.to(device))
                    for view, loc in zip(views, loc)
                ]

                # print(teacher_out.shape)
                loss = criterion(teacher_out, student_out, epoch=epoch)

                batch_loss += loss.detach().item()

        validation_loss = batch_loss / len(validation_loader)

        wandb.log({"val epoch_loss": validation_loss, "epoch": epoch})

        writer.add_scalar("validation loss", validation_loss, epoch)
        print(
            f"epoch: {epoch+1} / {num_epochs}, loss: {training_loss:.4f}, val_loss: {validation_loss:.4f}"
        )

        res_training_loss.append(training_loss)
        res_validation_loss.append(validation_loss)
        # res_validation_metric.append(validation_metric)
        # res_validation_dice.append(validation_dice)

        if epoch == data_info_train["epoch"] + 1:
            best_validation_loss = validation_loss  # /len(validation_loader)
            # best_validation_metric = validation_metric
            if epoch == 1:
                save_model(epoch, model, optimizer, criterion, models_dir)

        if (
            validation_loss < best_validation_loss
        ):  # edw to bazw megalutero alla thelei allagh
            best_validation_loss = validation_loss
            # best_validation_metric = validation_metric
            save_model(epoch, model, optimizer, criterion, models_dir)
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement > patience:
            break

        # scheduler.step(validation_loss)

    print("Finished Training")


def trainer_byol_loc(
    model,
    num_epochs,
    batch_size,
    spatial_size,
    num_sub_vol,
    learning_rate,
    criterion,
    optimizer,
    scheduler,
    device,
    models_dir,
    writer,
    init_epoch=0,
    data_info_train=None,
    data_info_val=None,
    wandb=None,
):

    ###scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, cooldown=2)

    print("device inside: ", device)
    # n_total_steps=len(train_loader) * num_sub_vol
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_total_steps, eta_min=10^(-7))

    res_training_loss = []
    res_validation_loss = []
    no_improvement = 0
    patience = 800
    conv1_grad_norm = 0
    conv2_grad_norm = 0
    for epoch in tqdm(range(init_epoch + 1, num_epochs)):  # bale tqdm na blepeis
        print(epoch)
        training_loss = 0
        batch_loss = 0
        step = 0
        model.train()
        if epoch % data_info_train["epoch_step"] == 0 or epoch == init_epoch + 1:
            if epoch != init_epoch + 1:
                del train_dataset
                del train_loader
                del validation_dataset
                del validation_loader
                gc.collect()

            data_info_train["cur_epoch"] = epoch % 3800
            data_info_val["cur_epoch"] = epoch % 3800
            train_dataset, train_loader = create_loader(data_info_train)

            validation_dataset, validation_loader = create_loader(data_info_val)
        momentum_val = cosine_schedule(epoch, num_epochs, 0.996, 1)
        for i, batch in enumerate(train_loader):
            step += 1
            views_temp, loc_views = batch  # [0]
            x0, x1 = views_temp
            x2, x3 = loc_views

            update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
            update_momentum(
                model.projection_head, model.projection_head_momentum, m=momentum_val
            )

            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)

            p0 = model(x0, x2)
            z0 = model.forward_momentum(x0, x2)

            p1 = model(x1, x3)
            z1 = model.forward_momentum(x1, x3)

            # z0 = nn.functional.normalize(z0, p=2, dim=1)
            # z1 = nn.functional.normalize(z1, p=2, dim=1)

            # loss = criterion(z0, z1)#
            loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))

            batch_loss += loss.detach().item()
            loss.backward()
            # Log gradients of the first and second convolutional layers
            conv1_grad_norm = model.backbone.layer1[0].conv2.weight.grad.norm().item()
            conv2_grad_norm = model.backbone.layer4[0].conv2.weight.grad.norm().item()
            wandb.log(
                {"conv1_grad_norm": conv1_grad_norm, "conv4_grad_norm": conv2_grad_norm}
            )

            max_grad_norm = 10

            # kleinw kai auto na doume
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            if scheduler != None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"current_lr": current_lr})
            optimizer.zero_grad()

        training_loss = batch_loss / len(train_loader)

        # writer.add_scalar('training loss', training_loss, epoch)
        # wandb.log({"train epoch_loss": training_loss})
        wandb.log({"train epoch_loss": training_loss, "epoch": epoch})
        # validation step
        validation_loss = 0
        model.eval()
        batch_loss = 0
        for i, batch in enumerate(validation_loader):
            with torch.no_grad():
                views_temp, loc_views = batch  # [0]
                x0, x1 = views_temp
                x2, x3 = loc_views
                update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
                update_momentum(
                    model.projection_head,
                    model.projection_head_momentum,
                    m=momentum_val,
                )
                x0 = x0.to(device)
                x1 = x1.to(device)
                x2 = x2.to(device)
                x3 = x3.to(device)

                p0 = model(x0, x2)
                z0 = model.forward_momentum(x0, x2)

                p1 = model(x1, x3)
                z1 = model.forward_momentum(x1, x3)

                # z0 = nn.functional.normalize(z0, p=2, dim=1)
                # z1 = nn.functional.normalize(z1, p=2, dim=1)

                # loss = criterion(z0, z1)#
                loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))

                batch_loss += loss.detach().item()

        validation_loss = batch_loss / len(validation_loader)

        wandb.log({"val epoch_loss": validation_loss, "epoch": epoch})

        writer.add_scalar("validation loss", validation_loss, epoch)
        print(
            f"epoch: {epoch+1} / {num_epochs}, loss: {training_loss:.4f}, val_loss: {validation_loss:.4f}"
        )

        res_training_loss.append(training_loss)
        res_validation_loss.append(validation_loss)
        # res_validation_metric.append(validation_metric)
        # res_validation_dice.append(validation_dice)

        if epoch == data_info_train["epoch"] + 1:
            best_validation_loss = validation_loss  # /len(validation_loader)
            # best_validation_metric = validation_metric
            if epoch == 1:
                save_model(epoch, model, optimizer, criterion, models_dir)

        if (
            validation_loss < best_validation_loss
        ):  # edw to bazw megalutero alla thelei allagh
            best_validation_loss = validation_loss
            # best_validation_metric = validation_metric
            save_model(epoch, model, optimizer, criterion, models_dir)
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement > patience:
            break

        # scheduler.step(validation_loss)

    print("Finished Training")


def update_ema(student_scalar_param, teacher_scalar_param, ema_momentum):

    # for param, ema_param in zip(student_model.parameters(), teacher_model.parameters()):
    #    ema_param.data.mul_(ema_momentum).add_((1 - ema_momentum) * param.data)
    #    #teacher_param.data.mul_(ema_momentum).add_(student_param.data, alpha=1 - ema_momentum)

    if student_scalar_param.requires_grad:
        teacher_scalar_param.data.mul_(ema_momentum).add_(
            student_scalar_param.data, alpha=1 - ema_momentum
        )


def trainer_moco_loc(
    model,
    num_epochs,
    batch_size,
    spatial_size,
    num_sub_vol,
    learning_rate,
    criterion,
    optimizer,
    scheduler,
    device,
    models_dir,
    writer,
    init_epoch=0,
    data_info_train=None,
    data_info_val=None,
    wandb=None,
):

    ###scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, cooldown=2)

    print("device inside: ", device)
    # n_total_steps=len(train_loader) * num_sub_vol
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_total_steps, eta_min=10^(-7))

    res_training_loss = []
    res_validation_loss = []
    no_improvement = 0
    patience = 800
    conv1_grad_norm = 0
    conv2_grad_norm = 0
    for epoch in tqdm(range(init_epoch + 1, num_epochs)):  # bale tqdm na blepeis
        print(epoch)
        training_loss = 0
        batch_loss = 0
        inv_loss_full = 0
        var_loss_full = 0
        cov_loss_full = 0
        step = 0
        model.train()
        if epoch % data_info_train["epoch_step"] == 0 or epoch == init_epoch + 1:
            if epoch != init_epoch + 1:
                del train_dataset
                del train_loader
                del validation_dataset
                del validation_loader
                gc.collect()

            """train_dataset=None
            train_loader=None
            validation_dataset=None
            validation_loader=None"""
            data_info_train["cur_epoch"] = epoch % 3800
            data_info_val["cur_epoch"] = epoch % 3800
            train_dataset, train_loader = create_loader(data_info_train)

            validation_dataset, validation_loader = create_loader(data_info_val)

        momentum_val = cosine_schedule(epoch, num_epochs, 0.996, 1)

        # momentum_val = cosine_schedule(epoch, num_epochs, 0.996, 1) to ekleisa twra
        for i, batch in enumerate(train_loader):
            step += 1
            views_temp, loc_views = batch  # [0]
            x0, x1 = views_temp
            x2, x3 = loc_views

            update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
            update_momentum(
                model.projection_head, model.projection_head_momentum, m=momentum_val
            )
            update_momentum(model.loc_fc, model.loc_fc_momentum, m=momentum_val)
            # update_momentum(model.trainable_parameters, model.trainable_parameters_momentum, m=momentum_val)
            update_momentum(model.fc1, model.fc1_momentum, m=momentum_val)

            # update with momentum the trainable parameters with custom function
            update_ema(
                model.trainable_parameters,
                model.trainable_parameters_momentum,
                momentum_val,
            )

            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            """print(x0.shape)
            print(x1.shape)
            print(x2.shape)
            print(x3.shape)
            sdf"""
            z0 = model(x0, x2)  # query
            z1 = model.forward_momentum(x1, x3)  # key

            """z0 = model(x0,x2)
            z1 = model(x1,x3)"""

            # z0 = nn.functional.normalize(z0, p=2, dim=1)
            # z1 = nn.functional.normalize(z1, p=2, dim=1)

            loss = criterion(z0, z1)  #

            batch_loss += loss.detach().item()
            loss.backward()
            # Log gradients of the first and second convolutional layers
            conv1_grad_norm = model.backbone.layer1[0].conv2.weight.grad.norm().item()
            conv2_grad_norm = model.backbone.layer4[0].conv2.weight.grad.norm().item()
            wandb.log(
                {"conv1_grad_norm": conv1_grad_norm, "conv4_grad_norm": conv2_grad_norm}
            )

            # edw prepei norm
            # model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
            max_grad_norm = 10

            # kleinw kai auto na doume
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            # to kalo einai

            # kalo autoooptimizer.zero_grad()

            # allagh gia scheduler edw
            if scheduler != None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"current_lr": current_lr})
            optimizer.zero_grad()

        training_loss = batch_loss / len(train_loader)

        # training_loss = training_loss/step

        writer.add_scalar("training loss", training_loss, epoch)
        # wandb.log({"train epoch_loss": training_loss})
        wandb.log({"train epoch_loss": training_loss, "epoch": epoch})

        # validation step
        validation_loss = 0
        val_steps = 0

        # validation_loss = valid(validation_loader,model,device,criterion)
        # add--------
        model.eval()
        # CosineEmbeddingLoss

        batch_loss = 0
        for i, batch in enumerate(validation_loader):
            with torch.no_grad():
                views_temp, loc_views = batch  # [0]
                x0, x1 = views_temp
                x2, x3 = loc_views

                update_momentum(
                    model.backbone, model.backbone_momentum, m=momentum_val
                )  # isws mia fora kai oxi kai sto validation
                update_momentum(
                    model.projection_head,
                    model.projection_head_momentum,
                    m=momentum_val,
                )
                update_momentum(model.loc_fc, model.loc_fc_momentum, m=momentum_val)
                # update_momentum(model.trainable_parameters, model.trainable_parameters_momentum, m=momentum_val)
                update_momentum(model.fc1, model.fc1_momentum, m=momentum_val)
                update_ema(
                    model.trainable_parameters,
                    model.trainable_parameters_momentum,
                    momentum_val,
                )

                x0 = x0.to(device)
                x1 = x1.to(device)
                x2 = x2.to(device)
                x3 = x3.to(device)
                """print(x0.shape)
                print(x1.shape)
                print(x2.shape)
                print(x3.shape)
                sdf"""
                z0 = model(x0, x2)  # query
                z1 = model.forward_momentum(x1, x3)  # key
                loss = criterion(z0, z1)

                batch_loss += loss.detach().item()
                # loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()

        validation_loss = batch_loss / len(validation_loader)
        # print(batch_loss)

        # here ----
        # val_steps = len(validation_loader)
        # validation_loss = validation_loss/(val_steps)
        # wandb.log({"val epoch_loss": validation_loss})
        wandb.log({"val epoch_loss": validation_loss, "epoch": epoch})

        writer.add_scalar("validation loss", validation_loss, epoch)
        print(
            f"epoch: {epoch+1} / {num_epochs}, loss: {training_loss:.4f}, val_loss: {validation_loss:.4f}"
        )

        res_training_loss.append(training_loss)
        res_validation_loss.append(validation_loss)
        # res_validation_metric.append(validation_metric)
        # res_validation_dice.append(validation_dice)

        if epoch == data_info_train["epoch"] + 1:
            best_validation_loss = validation_loss  # /len(validation_loader)
            # best_validation_metric = validation_metric
            if epoch == 1:
                save_model(epoch, model, optimizer, criterion, models_dir)

        if (
            validation_loss < best_validation_loss
        ):  # edw to bazw megalutero alla thelei allagh
            best_validation_loss = validation_loss
            # best_validation_metric = validation_metric
            save_model(epoch, model, optimizer, criterion, models_dir)
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement > patience:
            break

        # scheduler.step(validation_loss)

    print("Finished Training")
