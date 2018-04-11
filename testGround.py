import tensorflow as tf
import gym
print(gym.__version__)
# trst = [1, 2, 3]
# tt = [22, 3, 4, 5]
# print(*(trst+tt))
# mnist = input_data.read_data_sets
#
#
# for i in range(4):
#     print(i)class func(object):
#     def __init__(self):
#         self.a = 1
#         self.b = 2
#     def __call__(self, *args, **kwargs):
#         self.b = args(0)
#
# def cc(temp):
#     print(temp)
#
# def case(temp):
#     a = cc(temp)
#     return a
#
# case(2)
#
# a = 2
# a /=2
# print(a)
# for i in range(4,14):
#     print(i)
# a1 = [1, 2, 3, 4]
# a2 = [1, 2, 2, 6]
#
# for i, j in zip(a1, a2):
#     print(i)
#     print(j)
#     print("  ")
#
# temp = 1-1e-2
# print(temp)
#
# # import tensorflow as tf
# # print(tf.__version__)
#
# import tensorflow as tf
# with tf.variable_scope("foo") as foo_scope:
#     assert foo_scope.name == "foo"
#
# with tf.variable_scope("bar"):
#     with tf.variable_scope("baz") as other_scope:
#         assert other_scope.name == "bar/baz"
#         with tf.variable_scope(foo_scope) as foo_scope2:
#             assert foo_scope2.name == "foo"