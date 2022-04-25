from utils import train_cmd, weight_decays
from pytools.persistent_dict import PersistentDict
from time import sleep

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

class Locations:
  weight_decay = "logs/weight_decay_{wd}_seed_{seed}"
  dropout = "logs/dropout_{do}_seed_{seed}"
  slow_decoder = "logs/decoder_lr_{lr}_multiplier_{mlr}_seed_{seed}"
  esam = "logs/esam_rho_{rho}_beta_{beta}_seed_{seed}"

rule all:
  input:
    expand(Locations.weight_decay, wd=weight_decays, seed=[1,2,3]),
    # expand(Locations.dropout, do=[0, .1, .2, .3, .4, .5], seed=[1,2]),
    # expand(Locations.slow_decoder, lr=[1, .1, .01, .001], mlr=[.001, .01], seed=[1,2]),
    # expand(Locations.esam, rho=[.01, .05, .1, .5, 1], beta=[.5, 1], seed=[1,2]),


rule with_weight_decay:
  output: 
    logs = directory(Locations.weight_decay)
  run: 
    cmd = train_cmd(weight_decay=wildcards.wd, random_seed=wildcards.seed)
    run_on_free_gpu(cmd + f" --logdir={output.logs}")

rule with_dropout:
  output: 
    logs = directory(Locations.dropout)
  run: 
    cmd = train_cmd(dropout=wildcards.do, random_seed=wildcards.seed)
    run_on_free_gpu(cmd + f" --logdir={output.logs}")

rule with_slow_decoder:
  output: 
    logs = directory(Locations.slow_decoder)
  run: 
    cmd = train_cmd(decoder_lr=wildcards.lr, random_seed=wildcards.seed, lr_multiplier=wildcards.mlr)
    run_on_free_gpu(cmd + f" --logdir={output.logs}")

rule with_esam:
  output:
    logs = directory(Locations.esam)
  run:
    cmd = train_cmd(esam_rho=wildcards.rho, esam_beta=wildcards.beta, random_seed=wildcards.seed)
    run_on_free_gpu(cmd + f" --esam --logdir={output.logs}")
