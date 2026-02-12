import logging
from colorama import init, Fore, Style

# 初始化colorama
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """自定义彩色日志格式器"""

    # 定义颜色映射
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        # 获取原始格式
        message = super().format(record)

        # 根据日志级别添加颜色
        color = self.COLORS.get(record.levelname, '')
        if color:
            message = color + message + Style.RESET_ALL

        return message

# 配置日志


def setup_logging():
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # 创建彩色格式器
    formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler('model_optimizer.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            console_handler,
            file_handler
        ]
    )
