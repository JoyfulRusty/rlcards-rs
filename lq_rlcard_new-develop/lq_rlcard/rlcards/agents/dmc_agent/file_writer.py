# -*- coding: utf-8 -*-

import copy
import datetime
import csv
import json
import logging
import os
import time
from typing import Dict

import git


def gather_metadata() -> Dict:
    """
    收集元数据

    返回结果为字典(dict)
    """
    # 开始时间
    date_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    # 收集git元数据
    try:
        # 远程仓库
        repo = git.Repo(search_parent_directories=True)
        # 提交哈希
        git_sha = repo.commit().hexsha
        # 使用字典来存储git获取数据
        git_data = dict(
            commit=git_sha,
            branch=repo.active_branch.name,
            is_dirty=repo.is_dirty(),
            path=repo.git_dir)
    except git.InvalidGitRepositoryError:
        git_data = None

    # 收集slum元数据
    if 'SLURM_JOB_ID' in os.environ:  # slurm，一个 Linux服务器中的集群管理和作业调度系统
        slurm_env_keys = [k for k in os.environ if k.startswith('SLURM')]
        slurm_data = {}
        for k in slurm_env_keys:
            d_key = k.replace('SLURM_', '').replace('SLURMD_', '').lower()
            slurm_data[d_key] = os.environ[k]
    else:
        slurm_data = None

    # 返回字典存储到的数据
    return dict(
        date_start=date_start,
        date_end=None,
        successful=False,
        git=git_data,
        slurm=slurm_data,
        env=os.environ.copy(),
    )


class FileWriter:
    """ 写入文件 """
    def __init__(self,x_pid: str = None,xp_args: dict = None,root_dir: str = '~/palaas'):
        """ 初始化写入文件类属性参数 """
        if not x_pid:
            # 创建唯一id
            x_pid = '{proc}_{unixtime}'.format(proc=os.getpid(), unixtime=int(time.time()))
        self.x_pid = x_pid
        self._tick = 0

        # 获取元数据
        if xp_args is None:
            xp_args = {}
        self.metadata = gather_metadata()
        # 需要复制参数，否则当关闭文件编写器时
        # (并重写参数)可能有不可序列化的对象(或其他东西)
        self.metadata['args'] = copy.deepcopy(xp_args)
        self.metadata['x_pid'] = self.x_pid

        # 打印日志
        formatter = logging.Formatter('%(message)s')
        self._logger = logging.getLogger('palaas/out')

        # 到stdout处理程序
        handle = logging.StreamHandler()
        # 设置格式化
        handle.setFormatter(formatter)
        # 添加handler
        self._logger.addHandler(handle)
        # 日志等级
        self._logger.setLevel(logging.INFO)

        root_dir = os.path.expandvars(os.path.expanduser(root_dir))

        # 到文件处理程序
        self.basepath = os.path.join(root_dir, self.x_pid)

        if not os.path.exists(self.basepath):
            self._logger.info('Creating log directory: %s', self.basepath)
            os.makedirs(self.basepath, exist_ok=True)
        else:
            self._logger.info('Found log directory: %s', self.basepath)

        # 注意：删除最新版本，因为它在slurm上运行时会产生错误多个作业试图写入最新版本，
        # 但找不到添加“最新”作为符号链接，除非它存在并且不是符号链接

        # symlink = os.path.join(root_dir, 'latest')
        # if os.path.islink(symlink):
        #     os.remove(symlink)
        # if not os.path.exists(symlink):
        #     os.symlink(self.basepath, symlink)
        #     self._logger.info('Symlinked log directory: %s', symlink)

        self.paths = dict(
            msg='{base}/out.log'.format(base=self.basepath),
            logs='{base}/logs.csv'.format(base=self.basepath),
            fields='{base}/fields.csv'.format(base=self.basepath),
            meta='{base}/meta.json'.format(base=self.basepath),
        )

        self._logger.info('Saving arguments to %s', self.paths['meta'])
        if os.path.exists(self.paths['meta']):
            self._logger.warning('Path to meta file already exists.' 'Not overriding meta.')
        else:
            self._save_metadata()

        self._logger.info('Saving messages to %s', self.paths['msg'])
        if os.path.exists(self.paths['msg']):
            self._logger.warning('Path to message file already exists.' 'New data will be appended.')

        fhandle = logging.FileHandler(self.paths['msg'])
        fhandle.setFormatter(formatter)
        self._logger.addHandler(fhandle)

        self._logger.info('Saving logs data to %s', self.paths['logs'])
        self._logger.info('Saving logs\' fields to %s', self.paths['fields'])
        if os.path.exists(self.paths['logs']):
            self._logger.warning('Path to log file already exists.' 'New data will be appended.')
            with open(self.paths['fields'], 'r') as csvfile:
                reader = csv.reader(csvfile)
                self.fieldnames = list(reader)[0]
        else:
            self.fieldnames = ['_tick', '_time']

    def log(self, to_log: Dict, tick: int = None,verbose: bool = False) -> None:
        """ 日志 """
        if tick is not None:
            raise NotImplementedError
        else:
            to_log['_tick'] = self._tick
            self._tick += 1
        to_log['_time'] = time.time()

        old_len = len(self.fieldnames)
        for k in to_log:
            if k not in self.fieldnames:
                self.fieldnames.append(k)
        if old_len != len(self.fieldnames):
            with open(self.paths['fields'], 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.fieldnames)
            self._logger.info('Updated log fields: %s', self.fieldnames)

        if to_log['_tick'] == 0:
            with open(self.paths['logs'], 'a') as f:
                f.write('# %s\n' % ','.join(self.fieldnames))

        if verbose:
            self._logger.info('LOG | %s', ', '.join(['{}: {}'.format(k, to_log[k]) for k in sorted(to_log)]))

        with open(self.paths['logs'], 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(to_log)

    def close(self, successful: bool = True) -> None:
        """ 关闭 """
        self.metadata['date_end'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.metadata['successful'] = successful
        self._save_metadata()

    def _save_metadata(self) -> None:
        """ 保存元数据 """
        with open(self.paths['meta'], 'w') as jsonfile:
            json.dump(self.metadata, jsonfile, indent=4, sort_keys=True)