#!/usr/bin/env python

import numpy as np
# import matplotlib.pyplot as plt
import spotpy

from setups.cylinder_Bscan_2D_gprMax import spotpy_setup

results = []
rep = 10

sampler=spotpy.algorithms.sceua(spotpy_setup(), dbname='cylinder_Bscan_2D', dbformat='csv', parallel='mpi')
sampler.sample(rep, ngs=2)
results.append(sampler.getdata())

################################
# Loading and plotting results #
################################

# def find_min_max(spotpy_setup):
#     randompar=spotpy_setup.parameters()['random']
#     for i in range(1000):
#         randompar=np.column_stack((randompar,spotpy_setup.parameters()['random']))
#     return np.amin(randompar,axis=1),np.amax(randompar,axis=1)
#
# dbname = 'cylinder_Bscan_2D'
# results = spotpy.analyser.load_csv_results(dbname)
# s = spotpy_setup()
# min_vs, max_vs = find_min_max(s)
#
# fig = plt.figure(figsize=(16,16))
# numparams = s.parameters().size
#
# for p in range(numparams):
#     plt.subplot(numparams,2,2*p+1)
#     x = results['par'+s.parameters()['name'][p]]
#     for i in range(int(max(results['chain']))):
#         index=np.where(results['chain']==i)
#         plt.plot(x[index],'.')
#     plt.ylabel(s.parameters()['name'][p])
#     plt.ylim(min_vs[p],max_vs[p])
#
#     plt.subplot(numparams,2,2*p+2)
#     x = results['par'+s.parameters()['name'][p]][int(len(results)*0.5):]
#     normed_value = 1
#     hist, bins = np.histogram(x, bins=20, density=True)
#     widths = np.diff(bins)
#     hist *= normed_value
#     plt.bar(bins[:-1], hist, widths)
#     plt.ylabel(s.parameters()['name'][p])
#     plt.xlim(min_vs[p],max_vs[p])
#
# plt.show()
# # fig.savefig(dbname + '.png',dpi=300)
