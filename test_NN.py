import pickle

dummy_bkg_path = 'set3/bkg.pkl'
dummy_sig_path = 'set3/sig.pkl'
file = open(dummy_bkg_path, 'rb')
bkg_data = pickle.load(file)
file.close()

file = open(dummy_sig_path, 'rb')
sig_data = pickle.load(file)
file.close()

print('test show data')
print(sig_data)
print(bkg_data)
