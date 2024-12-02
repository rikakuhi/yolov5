from threading import Thread


class MyThread(Thread):
    """
    重写Thread函数，添加get_result()函数，获取线程处理结束的返回值
    """
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result
