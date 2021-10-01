#!/usr/bin/env python3

import os                                                                                     
import subprocess                                                                             
                                                                                              
bin_path = subprocess.check_output(['which', 'lux-ai-2021']).decode('ascii').strip()          
                                                                                              
while bin_path != os.path.realpath(bin_path):                                                 
    bin_path = os.path.realpath(bin_path)                                                     
                                                                                              
                                                                                              
print('bin_path', bin_path)
paths = bin_path.split(os.sep)                                                                
print('paths', paths)
#paths.pop()
#paths.pop()
modules_index = paths.index('node_modules')                                                   
modules_dir = os.sep.join(paths[:modules_index + 1])                                         
# modules_dir = os.path.join(os.sep.join(paths), 'lib/node_modules/@lux-ai/2021-challenge/node_modules') 
print('modules_dir', modules_dir)
agents_file = os.path.join(modules_dir, '@lux-ai', '2021-challenge', 'node_modules', 'dimensions-ai', 'lib', 'main', 'Agent', 'index.js')  
print('agents_dir', agents_file)
                                                                                              
with open(agents_file) as f:                                                                  
    content = f.read()                                                                        
                                                                                              
new_content = []                                                                              
for line in content.split('\n'):                                                              
    if 'SIGSTOP' in line:                                                                     
        line = '// %s' % line                                                                 
    if 'SIGKILL' in line:                                                                     
        line = line.replace('SIGKILL', 'SIGTERM')                                             
    new_content.append(line)                                                                  
                                                                                              
new_content = '\n'.join(new_content)                                                          
                                                                                              
with open(agents_file, 'w') as f:                                                             
    f.write(new_content)
