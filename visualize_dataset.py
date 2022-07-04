import tarfile
import matplotlib

path_to_data = '../../../ceph/cfalk/hc-3'
test_file = path_to_data + '/ec012ec.11/ec012ec.189.tar.gz'


tar = tarfile.open(test_file, 'r')
# tar.extractall(test_file)

# for info in tar:
# 	print(info.name)

tar.close()