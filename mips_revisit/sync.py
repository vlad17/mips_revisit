"""
Directory syncer, taken from track, which took it from ray.

https://github.com/richardliaw/track
"""

import distutils.spawn
import shutil
import subprocess
import time
from urllib.parse import urlparse
import hashlib

from . import log
from .utils import timeit

try:  # py3
    from shlex import quote
except ImportError:  # py2
    from pipes import quote


S3_PREFIX = "s3://"
GCS_PREFIX = "gs://"
ALLOWED_REMOTE_PREFIXES = (S3_PREFIX, GCS_PREFIX)

def simplehash(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def _check_remote(remote_dir):
    if not any(
        remote_dir.startswith(prefix) for prefix in ALLOWED_REMOTE_PREFIXES
    ):
        return False

    if remote_dir.startswith(
        S3_PREFIX
    ) and not distutils.spawn.find_executable("aws"):
        raise TrackError(
            "Upload uri starting with '{}' requires awscli tool"
            " to be installed".format(S3_PREFIX)
        )
    elif remote_dir.startswith(
        GCS_PREFIX
    ) and not distutils.spawn.find_executable("gsutil"):
        raise TrackError("Upload uri starting with '{}' requires gsutil tool")
    return True


def sync(src, dst, *args):
    with timeit() as t:
        _sync(src, dst, args)
    log.debug("sync from {} to {} in {:.2f} sec", src, dst, t.seconds)


def _sync(src, dst, args):
    if _check_remote(dst):
        remote_dir = dst
    elif _check_remote(src):
        remote_dir = src
    else:
        shutil.copy(src, dst)
        return

    local_to_remote_sync_cmd = None
    if remote_dir.startswith(S3_PREFIX):
        local_to_remote_sync_cmd = "aws s3 sync {} {} {}".format(
            quote(src), quote(dst), ' '.join(map(quote, args))
        )
    elif remote_dir.startswith(GCS_PREFIX):
        local_to_remote_sync_cmd = "gsutil rsync -r {} {} {}".format(
            quote(src), quote(dst), ' '.join(map(quote, args))
        )

    if local_to_remote_sync_cmd:
        final_cmd = local_to_remote_sync_cmd
        sync_process = subprocess.Popen(final_cmd, shell=True)
        ret = sync_process.wait()  # fail gracefully
        if ret != 0:
            log.info(
                "sync from {} to {} failed with return code {}", src, dst, ret
            )


def exists(remote):
    if not _check_remote(remote):
        return os.path.exists(f)
    if remote.startswith(S3_PREFIX):
        from boto3 import client

        c = client("s3")
        parsed = urlparse(remote, allow_fragments=False)
        bucket = parsed.netloc
        path = parsed.path
        while path.startswith("/"):
            path = path[1:]
        response = c.list_objects_v2(
            Bucket=bucket, Prefix=path, Delimiter="/", MaxKeys=1
        )
        for obj in response.get("Contents", []):
            if obj["Key"] == path:
                return True
        return False
    if remote.startswith(GCS_PREFIX):
        import tensorflow as tf

        return tf.gfile.Exists(f)
    raise ValueError("unhandled file type")
