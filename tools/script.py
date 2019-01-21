import os
import sys
import codecs
import re

# Global state offset
offset = 0

# print the usage of the program
def usage():
        print 'usage: python %s outfile infile1 infile2 [infile3...]' % (sys.argv[0])
        print '       outfile: output file'
        print '       infile1, infile2, infile3: input files'

# parse a NFA file
def process_nfa_file(fid, infile, outfile):
        global offset
        state_count = 0

        lines = infile.readlines()
        for line in lines:
                line = line.strip()

                # Skip the empty line
                if len(line) == 0:
                        continue
                
                # Skip the comment line starting with #
                if line.startswith('#'):
                        continue
                
                # First non-comment line should have # of states
                if state_count == 0:
                        state_count = int(line)
                        continue

                parts = line.split(':')
                if len(parts) != 2:
                        print 'Cannot parse the line %s' % line 
                        continue

                parts[0] = parts[0].strip()
                parts[1] = parts[1].strip()
                #print parts

                src_dst = parts[0].split('->')

                # A single state
                if len(src_dst) == 1:
                        # Handle an initial state
                        if 'initial' in parts[1]:
                                init_state = int(parts[0])
                                init_state = init_state + offset
                                outfile.write('%d : initial\n' % init_state)

                        # Handle an accepting state
                        elif 'accepting' in parts[1]:
                                accept_state = int(parts[0])
                                accept_state = accept_state + offset
                                outfile.write('%d : accepting %d\n' % (accept_state, fid))
                        
                        # Unknown cases
                        else:
                                print 'Cannot parse the line %s' % line

                # If this is a legal transition with a source and a destination state
                elif len(src_dst) == 2:
                        src_state = int(src_dst[0].strip())
                        dst_state = int(src_dst[1].strip())
                        src_state = src_state + offset
                        dst_state = dst_state + offset
                        outfile.write('%d -> %d : %s\n' % (src_state, dst_state, parts[1]))

                # Unknown cases
                else:
                        print 'Cannot parse the line %s' % line
                        
        offset = offset + state_count
        # print state_count

def atoi(text):
        return int(text) if text.isdigit() else text

def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ atoi(c) for c in re.split('(\d+)', text) ]

if __name__== "__main__":
        # We need a output file and AT LEAST two input files
        if len(sys.argv) < 4:
                usage()
                sys.exit(1)
        
        # Get the names of input file and output files
        outfile_name = sys.argv[1]
        infile_names = sys.argv[2:]

        # Sort the input file names based on rule ID (1, 2, .., 10, 11)
        infile_names.sort(key = natural_keys)
        #print 'Process %d input NFA files ' % len(infile_names)
        #print infile_names

        # Open the output file
        outfile = open(outfile_name, 'w')
        
        # Parse each NFA file
        fid = 1
        for infile_name in infile_names:
                outfile.write('# NFA file %s \n' % infile_name)
                # Note: Vzch uses UTF16 encoding to write the output
                infile = codecs.open(infile_name, 'r')
                # process the NFA file 
                process_nfa_file(fid, infile, outfile)
                infile.close()
                fid = fid + 1

        # Close the output file        
        outfile.close()

        # Open the output file again to write something at the beginning
        outfile = open(outfile_name, 'r+')
        old_content = outfile.read()
        outfile.seek(0) # go back to the beginning of the file

        outfile.write('# Merge %d NFA files\n' % len(infile_names))     # total # of merged NFA files  
        outfile.write(str(offset) + '\n') # total # of states

        outfile.write(old_content)
        outfile.close()
