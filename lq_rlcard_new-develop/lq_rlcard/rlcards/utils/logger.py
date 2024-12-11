# -*- coding: utf-8 -*-

import os
import csv


class Logger(object):
    """
    记录器保存运行结果并帮助根据结果绘制图表
    """

    def __init__(self, log_dir):
        """
        初始化绘图和日志文件代表的标签、图例和路径
        """
        self.log_dir = log_dir

    def __enter__(self):
        self.txt_path = os.path.join(self.log_dir, 'log.txt')
        self.csv_path = os.path.join(self.log_dir, 'performance.csv')
        self.fig_path = os.path.join(self.log_dir, 'fig.png')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.txt_file = open(self.txt_path, 'w')
        self.csv_file = open(self.csv_path, 'w')

        filenames = ['idx', 'episode', 'reward']

        self.writer = csv.DictWriter(self.csv_file, fieldnames=filenames)
        self.writer.writeheader()

        return self

    def log(self, text):
        """
        将文本写入日志文件，然后进行打印
        """
        self.txt_file.write(text + '\n')
        self.txt_file.flush()
        print(text)

    def log_performance(self, idx, episode, reward):
        """
        记录曲线中的点
        """
        self.writer.writerow({"idx": idx, 'episode': episode, 'reward': reward})
        print('')
        self.log('----------------------------------------')
        self.log('  idx          |  ' + str(idx))
        self.log('  episode      |  ' + str(episode))
        self.log('  reward       |  ' + str(reward))
        self.log('========================================')

    def __exit__(self, idx, value, traceback):
        """退出"""
        if self.txt_path is not None:
            self.txt_file.close()

        if self.csv_path is not None:
            self.csv_file.close()

        print('\nLogs saved in', self.log_dir)