#! /usr/bin/env python3
# -*- coding: utf-8 -**-


import ast
import math
import collections
import subprocess
from Crypto.Util import number
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

def diagonal(num_in, num_out):
  num_rot = num_in-1
  num_mult = num_in
  num_acc = num_in
  return num_rot, num_mult, num_acc


def next_prime(n, prime):
  for i in range(prime, 2**60):
    if ((i % (2*n)) == 1 and number.isPrime(i)):
      return i

def estimate(arch, effort='medium'):

  stats = seal_model(arch, effort)
  print("Inference time: ", stats["t"])
  print("Memory bandwidth: ", stats["m"])
  print("Network bandwidth: ", stats["n"])
  score = 0.75*score_time(stats['t'], 'high') + 0.25*score_network(stats["n"], 'medium')
  return score

def score_time(t, effort):
  # The input t is assumed to have a unit of microsecond
  if (effort == 'medium'):
    # 10 secs
    clip_threshold = 10**8
  if (effort == 'high'):
    # 1 secs
    clip_threshold = 10**7
  elif (effort == 'maximum'):
    # 0.1 secs
    clip_threshold = 10**6
  if (t > (clip_threshold)):
    t = clip_threshold

  return (1 - float(t)/float(clip_threshold))

def score_memory(m, effort):
  # The input t is assumed to have a unit of MBytes
  if (effort == 'medium'):
    # 10GBytes
    clip_threshold = 10**4
  if (effort == 'high'):
    # 10GBytes
    clip_threshold = 10**4
  elif (effort == 'maximum'):
    # 1GBytes
    clip_threshold = 10**3
  if (m > (clip_threshold)):
    t = clip_threshold

  return (1 - float(t)/float(clip_threshold))

def score_network(n, effort):
  # The input t is assumed to have a unit of MBytes
  if (effort == 'medium'):
    # 10GBytes
    clip_threshold = 10**4
  if (effort == 'high'):
    # 5GBytes
    clip_threshold = 5*10**3
  elif (effort == 'maximum'):
    # 1GBytes
    clip_threshold = 10**3
  if (n > (clip_threshold)):
    n = clip_threshold

  return (1 - float(n)/float(clip_threshold))


def seal_model(arch, effort):
  stats = {"t": 0, "m": 0, "n":0}
  all_conv_stats = {"t": 0, "m": 0, "n":0}
  all_gc_stats = {"t": 0, "m": 0, "n":0}
  base_gc_stat = load_gc_model('./model')
  for layer in arch:
    print(layer)
    if ("conv" in layer):
      parms = inst_he_parms(arch[layer], 'conv', effort)
      stats_conv = query_matrix_stats(parms)
      print('stat conv', stats_conv)
      add_stats(stats, stats_conv)
      add_stats(all_conv_stats, stats_conv)

    elif ('relu' in layer):
      print(arch[layer])
      if (arch[layer]['quant'] == 0):
        stats_gc['t'] = 0
        stats_gc['m'] = 0
        stats_gc['n'] = 0
        add_stats(stats, stats_gc)
      else:
        stats_gc = query_gc_stats(arch[layer]['quant'], arch[layer]['in_channel']*arch[layer]['in_height']*arch[layer]['in_width'], base_gc_stat)
        stats_gc['t'] = stats_gc['t']*1000
        add_stats(stats, stats_gc)
        add_stats(all_gc_stats, stats_gc)
      print('stat gc', stats_gc)

    elif ('square' in layer):
      parms = inst_square_parms(layer['quant'], layer['in_channel']*layer['in_height']*layer['in_width'])
      stats_sq = query_sq_stats(parms)
      add_stats(stats, stats_sq)

    elif ('avgpooling' in layer):
      base_avg_pool = 32346*(10**3)/(63**4)
      stats_pooling = {}
      stats_pooling['t'] = 0
      stats_pooling['m'] = 0
      stats_pooling['n'] = 0
      stats_pooling['t'] += arch[layer]['in_channel']*arch[layer]['in_height']*arch[layer]['in_width']*base_avg_pool

    elif ('fc' in layer):
      parms = inst_he_parms(arch[layer], 'fc', effort)
      stats_fc = query_matrix_stats(parms)
      add_stats(stats, stats_fc)

    else:
      print("error")
      exit(0)
  ouf = open('breakdown', 'w')
  print('stats gc', all_gc_stats, file=ouf)
  print('stats conv', all_conv_stats, file=ouf)
  print('stats pooling', stats_pooling, file=ouf)
  return stats

def inst_he_parms(curr_layer, layer_type, effort):
  if (layer_type == 'conv'):
    # Getting parameters
    in_quant = curr_layer['in_quant']
    fil_quant = curr_layer['fil_quant']
    in_height = curr_layer['in_height']
    in_width = curr_layer['in_width']
    fil_height = curr_layer['fil_height']
    fil_width = curr_layer['fil_width']
    plain_space = 2**(in_quant + fil_quant)
    dim = in_height * in_width

    # Frequency-domain homomorphic convolution
    num_rot = 0
    num_mult = 1
    num_acc = 0

    # Initialization
    success = False
    ciph_modulus_list = []
    n = 2**11
    prime_low_limit = 0
    while not success:
      total_bit_count = 0
      beta = (dim/n)
      ops_list = []
      # remake ops list to consider ciphertext packing and stride
      num_rot = math.ceil(num_rot * beta)
      num_mult = math.ceil(num_mult * beta)
      num_acc = math.ceil(num_acc * beta)
      ops_list.append(('num_rot', num_rot))
      ops_list.append(('num_mult', num_mult))
      ops_list.append(('num_acc', num_acc))

      # Lattice Parameter Generation
      ciph_modulus_list, plain_modulus, success = generate_modulus_dim(n, plain_space, dim, ops_list, prime_low_limit)
      for modulus in ciph_modulus_list:
        total_bit_count += math.ceil(math.log(modulus, 2))

      if (effort == 'maximum'):
        prime_low_limit = prime_low_limit + 1
      elif (effort == 'medium'):
        prime_low_limit = total_bit_count + 1
      else:
        prime_low_limit = total_bit_count + 1

      # Ternery secret HE params recommendation for 128-bit security
      if (total_bit_count <= 54):
        n = 2**11
      if (total_bit_count > 54) and (total_bit_count <= 109):
        n = 2**12
      if (total_bit_count > 109) and (total_bit_count <= 218):
        n = 2**13
      if (total_bit_count > 218) and (total_bit_count <= 438):
        n = 2**14
      if (total_bit_count > 438) and (total_bit_count <= 881):
        n = 2**15

    channel_per_ciph = n/dim
    channel_packing = (curr_layer['in_channel']*curr_layer['out_channel'])/channel_per_ciph
    ops_list[1] = ('num_mult', int(channel_packing))
    print('channel_packing', channel_packing)
    print("total_bit_count", total_bit_count)
    command_dict = {
        "plain_modulus": plain_modulus, "n": n, "ciph_modulus": ciph_modulus_list,
        "ops_list": ops_list, 'in_channel': curr_layer['in_channel'], 'out_channel': curr_layer['out_channel'],
        "channel_packing": channel_packing
    }
    return command_dict
  elif (layer_type == 'fc'):
    in_quant = curr_layer['in_quant']
    fil_quant = curr_layer['fil_quant']
    in_height = curr_layer['in_height']
    in_width = curr_layer['in_width']
    fil_height = curr_layer['fil_height']
    fil_width = curr_layer['fil_width']
    curr_layer['in_channel'] = 1
    curr_layer['out_channel'] = 1
    plain_space = 2**(in_quant + fil_quant)
    dim = in_height * in_width

    # Rough operation estimation
    num_rot, num_mult, num_acc = diagonal(in_width*in_height, fil_height)

    # Initialization
    success = False
    ciph_modulus_list = []
    n = 2**11
    prime_low_limit = 0
    while not success:
      total_bit_count = 0
      beta = (dim/n)
      ops_list = []
      # remake ops list to consider ciphertext packing and stride
      num_rot = math.ceil(num_rot / n) - 1
      num_mult = math.ceil(num_mult / n)
      num_acc = math.ceil(num_acc/n) + math.ceil(math.log(n/in_height, 2))
      ops_list.append(('num_rot', num_rot))
      ops_list.append(('num_mult', num_mult))
      ops_list.append(('num_acc', num_acc))

      # Lattice Parameter Generation
      ciph_modulus_list, plain_modulus, success = generate_modulus_dim(n, plain_space, dim, ops_list, prime_low_limit)
      for modulus in ciph_modulus_list:
        total_bit_count += math.ceil(math.log(modulus, 2))
      if (effort == 'maximum'):
        prime_low_limit = prime_low_limit + 1
      elif (effort == 'medium'):
        prime_low_limit = total_bit_count + 1
      else:
        prime_low_limit = total_bit_count + 1

      # Ternery secret HE params recommendation for 128-bit security
      if (total_bit_count <= 54):
        n = 2**11
      if (total_bit_count > 54) and (total_bit_count <= 109):
        n = 2**12
      if (total_bit_count > 109) and (total_bit_count <= 218):
        n = 2**13
      if (total_bit_count > 218) and (total_bit_count <= 438):
        n = 2**14
      if (total_bit_count > 438) and (total_bit_count <= 881):
        n = 2**15

    channel_per_ciph = n/dim
    channel_packing = (curr_layer['in_channel']*curr_layer['out_channel'])/channel_per_ciph
    command_dict = {
        "plain_modulus": plain_modulus, "n": n, "ciph_modulus": ciph_modulus_list,
        "ops_list": ops_list, 'in_channel': curr_layer['in_channel'], 'out_channel': curr_layer['out_channel'],
        "channel_packing": channel_packing
    }
    return command_dict

  else:
    return False


def generate_modulus_dim(n, plain_space, dim, ops_list, prime_low_limit):
  # error estimation
  plain_modulus = next_prime(n, plain_space)
  eta_init = 3.2*6 # standard deviation of 3.2 with 6sigma sample rejection
  w_relin = 2**6
  cumulative_eta = eta_init
  enable_rot = False
  for ops in ops_list:
    if ('rot' in ops[0]):
      cumulative_eta += eta_init*w_relin*math.sqrt(n)
      enable_rot = True
    elif ('mult' in ops[0]):
      cumulative_eta *= plain_modulus*math.sqrt(n) # ||v||u_{k}sqrt{n}
    elif ('acc' in ops[0]):
      init_cum_eta = cumulative_eta
      for i in range(ops[1]):
        cumulative_eta = math.sqrt((cumulative_eta**2) + (init_cum_eta**2)) # addition between two equal level ciphertexts
  # ciphertext modulus needs to tolerate maximum error with shifted plaintext
  # Execute noise estimator
  max_plaintext = plain_modulus * cumulative_eta
  plain_log = round(math.log(max_plaintext, 2))
  # generate ciphertext modulus that splits 2n-th cyclotomic field and tolerates the errors
  if (enable_rot):
    if (prime_low_limit == 0):
      sets_of_ciphs = plain_log//60 + 2
    else:
      sets_of_ciphs = prime_low_limit//60 + 2
  else:
    if (prime_low_limit == 0):
      sets_of_ciphs = plain_log//60 + 1
    else:
      sets_of_ciphs = prime_low_limit//60 + 1
  ciph_modulus_list = []
  for i in range(sets_of_ciphs):
    ciph_modulus = gen_prime(2*n, prime_low_limit % 59, plain_modulus, ciph_modulus_list)
    ciph_modulus_list.append(ciph_modulus)
  if (sanity_check(n, ciph_modulus_list)):
    command_dict = {
        "plain_modulus": plain_modulus, "n": n, "ciph_modulus": ciph_modulus_list,
        "ops_list": ops_list, 'in_channel': 1, 'out_channel': 1,
        "channel_packing": 1
    }
    command_list = generate_command(command_dict, "./error_est")
    result = subprocess.run(command_list, stdout=subprocess.PIPE)
    result_dec = result.stdout.decode('utf-8')
    success = False
    if (int(result_dec) >= 1):
      success = True
    return ciph_modulus_list, plain_modulus, success
  else:
    return ciph_modulus_list, plain_modulus, False

def sanity_check(n, ciph_modulus_list):
  total_bit_count = 0
  for modulus in ciph_modulus_list:
    total_bit_count += math.ceil(math.log(modulus, 2))
  # Ternery secret HE params recommendation for 128-bit security
  if (total_bit_count <= 54):
    if (n < 2**11):
      return False
  if (total_bit_count > 54) and (total_bit_count <= 109):
    if (n < 2**12):
      return False
  if (total_bit_count > 109) and (total_bit_count <= 218):
    if (n < 2**13):
      return False
  if (total_bit_count > 218) and (total_bit_count <= 438):
    if (n < 2**14):
      return False
  if (total_bit_count > 438) and (total_bit_count <= 881):
    if (n < 2**15):
      return False
  return True

def query_matrix_stats(parms):
  print('in channel', parms['in_channel'])
  print('out channel', parms['out_channel'])
  command_list = generate_command(parms, "./time_est")
  print(command_list)
  result = subprocess.run(command_list, stdout=subprocess.PIPE)
  stat_dict = {"t": 0.0, "m": 0.0, "n": 0.0}
  result_list = result.stdout.decode('utf-8').split()
  stat_dict['t'] = float(result_list[0])
  stat_dict['m'] = float(result_list[1])
  # traffic = calc_network(parms)
  stat_dict['n'] = float(result_list[2])
  return stat_dict

def generate_command(parms, executable):
  keyword = "{0:d}".format(parms['plain_modulus'])
  keyword += " {0:d}".format(parms['n'])
  parms['in_channel'] = int(parms['in_channel'])
  parms['out_channel'] = int(parms['out_channel'])
  keyword += " {0:d}-{1:d}-{2:f}".format(parms['in_channel'], parms['out_channel'], parms['channel_packing'])
  i = 0
  if (len(parms['ciph_modulus']) > 1):
    for mod in parms['ciph_modulus']:
      if (i == 0):
        keyword += " {0:d}-".format(mod)
      elif (i == len(parms['ciph_modulus']) - 1):
        keyword += "{0:d}".format(mod)
      else:
        keyword += "{0:d}-".format(mod)
      i += 1
  else:
    keyword += " {0:d}".format(parms['ciph_modulus'][0])
  for ops in parms['ops_list']:
    keyword += " {0:s}-{1:d}".format(ops[0], ops[1])
  command_list = []
  command_list.append(executable)
  command_list.extend(keyword.split())
  return command_list

def gen_prime(m, prime_low_limit, p, existing_list):
  start_prime = prime_low_limit
  for i in range(start_prime, 500):
    twopower = 2**i
    sqrtpw = int(math.sqrt(twopower))
    targetm = (twopower) % m
    if (targetm < 0):
      targetm += p

    for j in range(0, sqrtpw, m):
      pos_moded = j % p
      m_moded = j % m
      if (m_moded == targetm):
        gened_prime = 2**i-j+1
        if (number.isPrime(gened_prime)):
          if (gened_prime not in existing_list) and (gened_prime > p):
            return gened_prime
          break

def add_stats(stats, stats_add):
  stats['t'] += stats_add['t']
  stats['m'] += stats_add['m']
  stats['n'] += stats_add['n']

def query_gc_stats(quant, number, base_gc_stat):
  x = np.array([])
  x = np.append(x, number)
  x = x.reshape((-1, 1))
  return_dict = {}

  if (quant in base_gc_stat):
    model_dict = base_gc_stat[quant]
  else:
    model_dict = base_gc_stat[31]
  return_dict['t'] = model_dict['tt'].predict(x)[0]
  return_dict['m'] = 0
  return_dict['n'] = model_dict['tn'].predict(x)[0]
  return return_dict

def parse_gc_file(filename):
  inf = open(filename, 'r')
  input_dict = collections.OrderedDict()
  for line in inf:
    line_list = line.split()
    input_dict[line_list[0]] = line_list[1]


  number_list = []
  quant_list = []
  result_dict = {}
  for item in input_dict:
    num = int(item.split('-')[0])
    quant = int(item.split('-')[1])
    number_list.append(num)
    if (quant not in quant_list):
      quant_list.append(quant)
  
  print(quant_list)
  for quant in quant_list:
    exist_num_list = []
    x = np.array([])
    scaled_ot_perf_list = np.array([])
    scaled_tt_perf_list = np.array([])
    scaled_on_perf_list = np.array([])
    scaled_tn_perf_list = np.array([])
    for num in number_list:
      if num not in exist_num_list:
        x = np.append(x, num)
        stats = input_dict[str(num)+'-'+str(quant)]
        online_time = float(stats.split('-')[0])
        total_time = float(stats.split('-')[1])
        online_network = float(stats.split('-')[2])
        total_network = float(stats.split('-')[3])
        scaled_ot_perf_list = np.append(scaled_ot_perf_list, online_time)
        scaled_tt_perf_list = np.append(scaled_tt_perf_list, total_time)
        scaled_on_perf_list = np.append(scaled_on_perf_list, online_network)
        scaled_tn_perf_list = np.append(scaled_tn_perf_list, total_network)
        exist_num_list.append(num)

    x = x.reshape((-1, 1))

    result_dict[quant] = {}
    # Fit each component
    # Fit online time
    model = LinearRegression()
    model.fit(x, scaled_ot_perf_list)
    result_dict[quant]['ot'] = model
    # Fit total time
    model = LinearRegression()
    model.fit(x, scaled_tt_perf_list)
    result_dict[quant]['tt'] = model
    # Fit online network
    model = LinearRegression()
    model.fit(x, scaled_on_perf_list)
    result_dict[quant]['on'] = model
    # Fit total network
    model = LinearRegression()
    model.fit(x, scaled_on_perf_list)
    result_dict[quant]['tn'] = model

  return result_dict

def load_gc_model(folder_path):
  result_dict = {}
  for quant in range (2, 32):
    result_dict[quant] = {}
    # ot
    model = joblib.load('./model/model_'+str(quant)+'_ot')
    result_dict[quant]['ot'] = model
    # tt
    model = joblib.load('./model/model_'+str(quant)+'_tt')
    result_dict[quant]['tt'] = model
    # on
    model = joblib.load('./model/model_'+str(quant)+'_on')
    result_dict[quant]['on'] = model
    # tn
    model = joblib.load('./model/model_'+str(quant)+'_tn')
    result_dict[quant]['tn'] = model
  return result_dict
