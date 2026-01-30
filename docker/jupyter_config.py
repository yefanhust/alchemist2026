# JupyterLab 配置

c = get_config()

# 基础配置
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True

# 安全配置（开发环境）
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.allow_origin = '*'

# 目录配置
c.ServerApp.root_dir = '/workspace'
c.ServerApp.notebook_dir = '/workspace/notebooks'

# 终端配置
c.ServerApp.terminado_settings = {'shell_command': ['/bin/bash']}

# 扩展配置
c.LabApp.extensions_in_dev_mode = True
