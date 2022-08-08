from utils import exec_cmd, get_model_path

exp = 'gmn_icml_*'

exec_cmd('scp -r yba@scai1.cs.ucla.edu:/local2/yba/GraphMatching/model/OurMCS/logs/{} '
         '{}/OurMCS/logs'.format(exp, get_model_path()))
print('done')
