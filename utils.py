"""
此文件放置一些辅助函数
"""


def print_time(start_time, end_time, des="整个过程"):
    spend_time = end_time - start_time
    if spend_time < 60:
        print(des + "耗时 %.2f 秒" % spend_time)
    elif 60 <= spend_time < 3600:
        print(des + "耗时 %.2f 分钟" % (spend_time / 60))
    elif spend_time >= 3600:
        print(des + "耗时 %.2f 小时" % (spend_time / 3600))


class ParameterError(Exception):
    def __init__(self, error_info):
        super(ParameterError, self).__init__()
        self.error_info = error_info

    def __str__(self):
        return self.error_info
