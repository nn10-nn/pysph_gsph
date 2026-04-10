#!/bin/bash
# -------------------------------------
# 本地 Codex 自动提交 & push
# -------------------------------------

WORKDIR="/c/Users/nan/pysph_gsph"
BRANCH="srhd-gsph-dev"

echo "进入工作目录: $WORKDIR"
cd "$WORKDIR" || exit 1

# 添加所有修改
git add .

# 自动 commit，带时间戳
COMMIT_MSG="Codex automatic commit $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$COMMIT_MSG"

# push 到远程分支
git push origin "$BRANCH"

echo "本地修改已提交并推送到 GitHub 分支 $BRANCH"
