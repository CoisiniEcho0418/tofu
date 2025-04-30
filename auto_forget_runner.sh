#!/bin/bash

# 原始配置文件路径（根据你的描述）
CONFIG_FILE="/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/config/forget/forget_phi_wmdp.yaml"
BACKUP_FILE="${CONFIG_FILE}.bak"  # 备份文件路径
LOG_FILE="/home/wxy/wxy_workspace/LLM_unlearn/tofu/run.log"  # 日志文件路径
FORGET_LOSS_VALUES=("RMU" "ME")   # 需要测试的 forget_loss 值列表
PYTHON_SCRIPT="/home/wxy/wxy_workspace/LLM_unlearn/tofu/forget.py"  # forget.py 绝对路径
TIMEOUT_DURATION="6h"             # 单任务最大运行时间（6小时）
# ---------------------------------------------------------

# 初始化日志
echo "===== 脚本开始执行 $(date) =====" > $LOG_FILE

# 创建备份文件
cp "$CONFIG_FILE" "$BACKUP_FILE" || {
    echo "[错误] 无法创建配置文件备份!" | tee -a $LOG_FILE
    exit 1
}

# 定义清理函数（用于异常捕获）
cleanup() {
    if [ $ERROR_FLAG -ne 0 ]; then
        echo "[异常] 恢复原始配置文件..." | tee -a $LOG_FILE
        mv "$BACKUP_FILE" "$CONFIG_FILE"
    fi
    rm -f tmp_config.yaml 2>/dev/null
}

# 注册异常捕获
trap cleanup EXIT SIGINT SIGTERM

# 主循环
ERROR_FLAG=0
for loss_value in "${FORGET_LOSS_VALUES[@]}"; do
    echo "当前处理参数: $loss_value [$(date)]" | tee -a $LOG_FILE
    
    # 生成临时配置文件（YAML格式兼容）
    sed "s/^forget_loss:.*/forget_loss: $loss_value/" "$BACKUP_FILE" > tmp_config.yaml || {
        echo "[错误] 配置文件修改失败!" | tee -a $LOG_FILE
        ERROR_FLAG=2
        exit 2
    }
    
    # 替换原配置文件
    mv tmp_config.yaml "$CONFIG_FILE" || {
        echo "[错误] 配置文件替换失败!" | tee -a $LOG_FILE
        ERROR_FLAG=3
        exit 3
    }
    
    # 执行训练程序（带超时和错误捕获）
    echo "启动训练任务: $loss_value" | tee -a $LOG_FILE
    timeout $TIMEOUT_DURATION python "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    
    # 检查执行结果
    if [ $EXIT_CODE -eq 0 ]; then
        echo "执行成功: $loss_value [$(date)]" | tee -a $LOG_FILE
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "[超时] 程序执行超过 $TIMEOUT_DURATION 被终止: $loss_value" | tee -a $LOG_FILE
        ERROR_FLAG=4
        break
    else
        echo "[错误] 非零退出码 $EXIT_CODE: $loss_value" | tee -a $LOG_FILE
        ERROR_FLAG=5
        break
    fi
    
    # 添加安全间隔（避免资源冲突）
    sleep 30
done

# 最终清理
cleanup

# 根据错误标志退出
if [ $ERROR_FLAG -ne 0 ]; then
    echo "===== 脚本异常退出 (错误码 $ERROR_FLAG) =====" | tee -a $LOG_FILE
    exit $ERROR_FLAG
else
    echo "===== 脚本全部执行完成 =====" | tee -a $LOG_FILE
    exit 0
fi