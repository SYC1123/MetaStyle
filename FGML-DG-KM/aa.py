
def train_one_epoch_1(model, optimizer, epoch, sourcetraindataloader, task_id, traindataloder=None):
    model.train()
    loss_list = []
    dice_list = []
    # 写入log.txt
    with open('log.txt', 'a') as f:
        f.write(f"当前开始训练第{task_id}个任务\n")

    if traindataloder is None:  # 当前只有源域
        for i, (img0, label) in enumerate(
                tqdm(sourcetraindataloader, total=len(sourcetraindataloader), desc=f'Epoch {epoch + 1}')):
            # TODO: 检查梯度
            optimizer.zero_grad()
            # 获取源域特征
            source_data = img0.to(device)
            label = label.to(device)
            source_features, segmentation_output = model(source_data)
            source_mean, source_std = compute_mean_and_std(source_features)  # 得到源域的均值和方差

            total_style_loss = 0.0

            segmentation_loss = criterion(segmentation_output, label)
            total_loss = segmentation_loss.clone()

            # 该部分是meta-test
            model.mode = 'meta-test'
            source_features, segmentation_output = model(source_data, mode='meta-test', meta_loss=total_loss)
            meta_loss = criterion(segmentation_output, label)

            dice_score = dice(segmentation_output, label)
            dice_list.append(dice_score.cpu().detach().numpy())

            loss_list.append(meta_loss.item())

            meta_loss.backward()
            optimizer.step()
    else:  # 当前有源域和目标域
        for source, target in zip(sourcetraindataloader, traindataloder):
            optimizer.zero_grad()
            img0, label = source
            img1, label1 = target
            # # 可视化img0   img1
            # plt.imshow(img0[0][0].cpu().detach().numpy(), cmap='gray')
            # plt.title('img0')
            # plt.show()
            # plt.imshow(img1[0][0].cpu().detach().numpy(), cmap='gray')
            # plt.title('img1')
            # plt.show()
            # 获取源域特征
            source_data = img0.to(device)
            label = label.to(device)
            label1 = label1.to(device)
            with torch.no_grad():
                source_features, _ = model(source_data)
                # # 可视化特征
                # plt.imshow(source_features[0][0].cpu().detach().numpy())
                # plt.title('source_features')
                # plt.show()

                source_mean, source_std = compute_mean_and_std(source_features)  # 得到源域的均值和方差
                # print('source_mean:', source_mean)
                # print('source_std:', source_std)
            #
            # total_style_loss = 0.0
            # for target_domain:
            target_data = img1.to(device)
            target_features, segementation_output = model(target_data)
            # # 可视化特征
            # plt.imshow(target_features[0][0].cpu().detach().numpy())
            # plt.title('target_features')
            # plt.show()

            target_mean, target_std = compute_mean_and_std(target_features)
            # task_id = 1,2 是相似域，3,4,5是不相似域
            if task_id == 1 or task_id == 2:
                labels = torch.ones(source_features.size(0), 1).to(device)
            else:
                labels = torch.zeros(source_features.size(0), 1).to(device)

            # print('target_mean:', target_mean)
            # print('target_std:', target_std)

            # # 动态计算权重
            # # 当前域和源域之间的偏差权重
            dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)
            # print('dynamic_weight:', dynamic_weight)
            #
            # # 计算风格对齐损失
            # style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)
            # print('style_loss:', style_loss)
            # total_style_loss += style_loss

            # 对比损失
            style_loss = criterion1(source_features, target_features, labels)
            # print('style_loss:', style_loss)

            # 分割任务损失
            segmentation_loss = criterion(segementation_output, label1)
            # print('segmentation_loss:', segmentation_loss)

            # 总损失
            total_loss = (1-dynamic_weight)*segmentation_loss + dynamic_weight*style_loss


            # 该部分是meta-test
            model.mode = 'meta-test'
            source_features, segmentation_output = model(target_data, mode='meta-test', meta_loss=total_loss)
            meta_loss = criterion(segmentation_output, label1)

            dice_score = dice(segmentation_output, label1)
            dice_list.append(dice_score.cpu().detach().numpy())

            loss_list.append(meta_loss.item())
            meta_loss.backward()
            optimizer.step()

    loss = np.mean(loss_list)
    dice_score = np.mean(dice_list)
    if (epoch + 1) % 10 == 0:
        # 保存模型
        torch.save(model.state_dict(), f'./meta_style_unet_{epoch + 1}.pth')
    # 写入log.txt
    source_metric = compute_source_metric(model, testdataloder)
    print('source_metric:', source_metric)
    with open('log.txt', 'a') as f:
        f.write(f"train—Epoch {epoch + 1}, Total Loss: {loss:.4f}, Dice Score: {dice_score:.4f}\n")