from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./logs")

# writer.add_scalar(tag, scalar_value, global_step)
# tag 名称, scalar_value y轴值, global_step x轴值
for step in range(100):
    writer.add_scalar("scaler/y=x", step, step)
    writer.add_scalar("scaler/y=x^2", step ** 2, step)

writer.close()