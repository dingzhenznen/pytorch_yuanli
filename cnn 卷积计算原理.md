CNN 的卷积就是“滑动窗口内积”：卷积核每滑到一个位置，就和对应的局部输入区域做一次逐元素相乘再求和。


› 逐元素相乘再求和  与 torch.dot(x, y) 和 torch.matmul(x, y) 都不相同是吗'


• 要分情况。

  逐元素相乘再求和本质上就是“内积 / 点积”：

  (x * y).sum()

  如果 x 和 y 是一维向量，那么它和 torch.dot(x, y) 是一样的。

  x = torch.tensor([1, 2, 3])
  y = torch.tensor([4, 5, 6])

  (x * y).sum()
  # 1*4 + 2*5 + 3*6 = 32

  torch.dot(x, y)
  # 32

  所以：

  一维向量：逐元素相乘再求和 = torch.dot

  如果 x 和 y 是二维矩阵，比如 2x2：

  x = torch.tensor([
      [1, 2],
      [3, 4],
  ])

  y = torch.tensor([
      [5, 6],
      [7, 8],
  ])

  逐元素相乘再求和是：

  (x * y).sum()

  计算：

  1*5 + 2*6 + 3*7 + 4*8 = 70

  得到一个数字。

  这时它等价于：

  torch.dot(x.flatten(), y.flatten())

  但不等价于：

  torch.matmul(x, y)

  因为矩阵乘法是：

  x @ y

  结果是一个 2x2 矩阵：

  [
    [1*5 + 2*7, 1*6 + 2*8],
    [3*5 + 4*7, 3*6 + 4*8],
  ]
  =
  [
    [19, 22],
    [43, 50],
  ]

  所以总结：

  (x * y).sum()

  和：

  torch.dot(x, y)

  关系是：

  如果 x, y 是一维向量：一样
  如果 x, y 是二维矩阵：torch.dot 不能直接用，需要 flatten 后才一样

  和：

  torch.matmul(x, y)

  关系是：

  通常不一样
  matmul 是矩阵乘法
  逐元素相乘再求和是整体内积

  在 CNN 里，一个局部图像块和卷积核都是小矩阵：

  patch.shape = (2, 2)
  kernel.shape = (2, 2)

  卷积那一步是：

  (patch * kernel).sum()

  也可以理解为：

  torch.dot(patch.flatten(), kernel.flatten())

  它得到的是一个数字，作为输出特征图上的一个位置。