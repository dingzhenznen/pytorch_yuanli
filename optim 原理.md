zero_grad(set_to_none=True)：清空参数的梯度缓存。设置为 True 时会将梯度设为 None，比设为 0 更节省显存
step()：执行单次参数更新，根据梯度和学习率更新模型参数
state_dict()：获取优化器状态字典，可用于保存检查点


 optimizer.zero_grad()
 optimizer.step 简化了 autograd 的步骤