#!/bin/bash
# filepath: /home/HD/course/pe_exp_student/matrix_sim.sh

# Create matrix_sim directory
mkdir -p /home/HD/course/pe_exp_student/sim_iverilog/matrix_sim

# Enter simulation directory
cd /home/HD/course/pe_exp_student/sim_iverilog

# Compile and run matrix PE simulation - 只使用matrix_pe目录下的文件
iverilog -o test_matrix -I../src/matrix_pe tb_top_2.v ../src/matrix_pe/matrix_pe.v ../src/matrix_pe/parallel_pe.v ../src/matrix_pe/pe_mult.v ../src/matrix_pe/pe_acc.v
./test_matrix

# Check if waveform file is generated
if [ -f "matrix_sim/matrix_pe.vcd" ]; then
    echo "Waveform file generated: matrix_sim/matrix_pe.vcd"
    ls -la matrix_sim/
else
    echo "Failed to generate waveform file"
fi