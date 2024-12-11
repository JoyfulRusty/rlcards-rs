# -*- coding: utf-8 -*-

import os
import git
import csv
import copy
import json
import time
import logging
import datetime
from typing import Dict


def gather_metadata() -> Dict:
    """
    收集git元数据
    """
    date_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.commit().hexsha
        git_data = dict(
            commit=git_sha,
            branch=repo.active_branch.name,
            is_dirty=repo.is_dirty(),
            path=repo.git_dir,
        )
    except git.InvalidGitRepositoryError:
        git_data = None
    # 收集slurm元数据
    if 'SLURM_JOB_ID' in os.environ:
        slurm_env_keys = [k for k in os.environ if k.startswith('SLURM')]
        slurm_data = {}
        for k in slurm_env_keys:
            d_key = k.replace('SLURM_', '').replace('SLURMD_', '').lower()
            slurm_data[d_key] = os.environ[k]
    else:
        slurm_data = None
    return dict(
        date_start=date_start,
        date_end=None,
        successful=False,
        git=git_data,
        slurm=slurm_data,
        env=os.environ.copy(),
    )


class FileWriter:
    def __init__(self, x_pid: str = None, xp_args: dict = None, root_dir: str = '~/palaas'):
        if not x_pid:
            # 创建唯一ID
            x_pid = '{proc}_{unixtime}'.format(proc=os.getpid(), unixtime=int(time.time()))
        self.x_pid = x_pid
        self._tick = 0

        # 元数据收集
        if xp_args is None:
            xp_args = {}
        self.metadata = gather_metadata()
        # 需要复制参数，否则当关闭文件编写器(并重写参数)时
        # 可能会得到不可序列化的对象或其他讨厌的东西
        self.metadata['args'] = copy.deepcopy(xp_args)
        self.metadata['x_pid'] = self.x_pid

        formatter = logging.Formatter('%(message)s')
        self._logger = logging.getLogger('palaas/out')

        # 日志输出
        stream_handle = logging.StreamHandler()
        stream_handle.setFormatter(formatter)
        self._logger.addHandler(stream_handle)
        self._logger.setLevel(logging.INFO)

        root_dir = os.path.expandvars(os.path.expanduser(root_dir))
        # 文件输出
        self.base_path = os.path.join(root_dir, self.x_pid)

        if not os.path.exists(self.base_path):
            self._logger.info('Creating log directory: %s', self.base_path)
            os.makedirs(self.base_path, exist_ok=True)
        else:
            self._logger.info('Found log directory: %s', self.base_path)
            
        self.paths = dict(
            msg='{base}/out.log'.format(base=self.base_path),
            logs='{base}/logs.csv'.format(base=self.base_path),
            fields='{base}/fields.csv'.format(base=self.base_path),
            meta='{base}/meta.json'.format(base=self.base_path),
        )

        self._logger.info('Saving arguments to %s', self.paths['meta'])
        if os.path.exists(self.paths['meta']):
            self._logger.warning('Path to meta file already exists. ' 'Not overriding meta.')
        else:
            self._save_metadata()
        self._logger.info('Saving messages to %s', self.paths['msg'])
        if os.path.exists(self.paths['msg']):
            self._logger.warning('Path to message file already exists. ' 'New data will be appended.')
        file_handle = logging.FileHandler(self.paths['msg'])
        file_handle.setFormatter(formatter)
        self._logger.addHandler(file_handle)

        self._logger.info('Saving logs data to %s', self.paths['logs'])
        self._logger.info('Saving logs\' fields to %s', self.paths['fields'])
        if os.path.exists(self.paths['logs']):
            self._logger.warning('Path to log file already exists. ' 'New data will be appended.')
            with open(self.paths['fields'], 'r') as csv_file:
                reader = csv.reader(csv_file)
                self.fieldnames = list(reader)[0]
        else:
            self.fieldnames = ['_tick', '_time']

    def log(self, to_log: Dict, tick: int = None, verbose: bool = False) -> None:
        """
        日志
        """
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
            with open(self.paths['fields'], 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self.fieldnames)
            self._logger.info('Updated log fields: %s', self.fieldnames)
        if to_log['_tick'] == 0:
            with open(self.paths['logs'], 'a') as f:
                f.write('# %s\n' % ','.join(self.fieldnames))
        if verbose:
            self._logger.info('LOG | %s', ', '.join(
                ['{}: {}'.format(k, to_log[k]) for k in sorted(to_log)]))
        with open(self.paths['logs'], 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(to_log)

    def close(self, successful: bool = True) -> None:
        """
        关闭流程
        """
        self.metadata['date_end'] = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S.%f')
        self.metadata['successful'] = successful
        self._save_metadata()

    def _save_metadata(self) -> None:
        """
        保存元数据
        """
        with open(self.paths['meta'], 'w') as jsonfile:
            json.dump(self.metadata, jsonfile, indent=4, sort_keys=True)