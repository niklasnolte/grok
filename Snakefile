from utils import train_cmd
from pytools.persistent_dict import PersistentDict

gpu_jobs = PersistentDict("jobs")

onstart:
  gpu_jobs.store("gpu0", 0)
  gpu_jobs.store("gpu1", 0)

def run_on_free_gpu(cmd):
  while True:
    job_counts = [gpu_jobs.fetch("gpu0"), gpu_jobs.fetch("gpu1")]
    print(job_counts)
    if job_counts[0] >= 3 and job_counts[1] >= 3: # at most 3 jobs per gpu
      sleep(15)
      continue
    cuda_id = 0 if job_counts[0] <= job_counts[1] else 1
    gpu_jobs.store(f"gpu{cuda_id}", job_counts[cuda_id] + 1)
    print(f"running on GPU {cuda_id}")
    shell(cmd + f" --gpu {cuda_id}")
    gpu_jobs.store(f"gpu{cuda_id}", gpu_jobs.fetch(f"gpu{cuda_id}") - 1)
    break

rule all:
  input:
    expand("logs/weight_decay_{wd}", wd=[0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]),
    expand("logs/dropout_{do}", do=[0, .1, .2, .3, .5]),
    expand("logs/decoder_lr_{lr}", lr=[1, .1, .01, .001]),
    expand("logs/esam_rho_{rho}_beta_{beta}_gamma_{gamma}", rho=[.01, .05, .1, .5, 1], beta=[.5, 1, 2], gamma=[.5, 1, 2]),


rule with_weight_decay:
  output: 
    logs = directory("logs/weight_decay_{wd}")
  run: 
    cmd = train_cmd(weight_decay=wildcards.wd)
    run_on_free_gpu(cmd + f" --logdir={output.logs}")

rule with_dropout:
  output: 
    logs = directory("logs/dropout_{do}")
  run: 
    cmd = train_cmd(dropout=wildcards.do)
    run_on_free_gpu(cmd + f" --logdir={output.logs}")

rule with_slow_decoder:
  output: 
    logs = directory("logs/decoder_lr_{lr}")
  run: 
    cmd = train_cmd(decoder_lr=wildcards.lr)
    run_on_free_gpu(cmd + f" --logdir={output.logs}")

rule with_esam:
  output:
    logs = directory("logs/esam_rho_{rho}_beta_{beta}_gamma_{gamma}")
  run:
    cmd = train_cmd(esam_rho=wildcards.rho, esam_beta=wildcards.beta, esam_gamma=wildcards.gamma)
    run_on_free_gpu(cmd + f" --esam --logdir={output.logs}")
