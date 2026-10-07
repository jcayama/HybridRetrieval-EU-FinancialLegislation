from multiprocessing import cpu_count
from key_config import *

# Socket Path

bind = 'unix:/var/www/html/web/HybridRetrieval-EU-FinancialLegislation/gunicorn.sock'

pythonpath = '/var/www/html/web/HybridRetrieval-EU-FinancialLegislation/.venv/bin/python'

# Worker Options

workers = cpu_count() + 1

worker_class = 'uvicorn.workers.UvicornWorker'



# Logging Options

loglevel = 'debug'

accesslog = '/var/www/html/web/HybridRetrieval-EU-FinancialLegislation/logs/access_log'

errorlog =  '/var/www/html/web/HybridRetrieval-EU-FinancialLegislation/logs/error_log'