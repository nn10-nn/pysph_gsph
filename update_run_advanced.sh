#!/bin/bash
# ---------------------------
# 服务器自动同步 GitHub & 运行多算例
# ---------------------------

REPO_PATH="$HOME/pysph-srhd-dev"
BRANCH="srhd-gsph-dev"

# 要运行的算例列表
EXAMPLES=(
    "examples/gas_dynamics/blastwave.py"
    "examples/gas_dynamics/sodShockTube_rel.py"
    # 可以添加更多算例
)

# 日志文件
LOGFILE="$REPO_PATH/run_log.txt"

echo "--------------------------------------------" >> "$LOGFILE"
echo "Run started at $(date)" >> "$LOGFILE"

# 进入仓库目录
cd "$REPO_PATH" || exit 1

# 保存当前 commit，以便出错回退
LAST_COMMIT=$(git rev-parse HEAD)

# 拉取最新代码
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

# 激活虚拟环境（如果有）
if [ -d ".venv" ]; then
    echo "激活虚拟环境..." >> "$LOGFILE"
    source .venv/bin/activate
fi

# 循环运行算例
for SCRIPT in "${EXAMPLES[@]}"; do
    echo "运行 $SCRIPT ..." | tee -a "$LOGFILE"
    python "$SCRIPT" >> "$LOGFILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "算例 $SCRIPT 运行失败，回退到上一个 commit" | tee -a "$LOGFILE"
        git reset --hard "$LAST_COMMIT"
        exit 1
    fi
done

echo "所有算例运行完成成功！" | tee -a "$LOGFILE"
echo "Run finished at $(date)" >> "$LOGFILE"
