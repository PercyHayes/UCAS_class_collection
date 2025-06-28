#!/bin/bash
# filepath: /home/HD/course/pe_exp_student/parallel_sim.sh

# Create parallel_sim directory
mkdir -p /home/HD/course/pe_exp_student/sim_iverilog/parallel_sim

# Enter simulation directory
cd /home/HD/course/pe_exp_student/sim_iverilog

# Compile and run parallel PE simulation
iverilog -o test_parallel -I../src/parallel_pe tb_top_1.v ../src/parallel_pe/parallel_pe.v ../src/parallel_pe/pe_mult.v ../src/parallel_pe/pe_acc.v
./test_parallel

# Check if waveform file is generated
if [ -f "parallel_sim/parallel_pe.vcd" ]; then
    echo "Waveform file generated: parallel_sim/parallel_pe.vcd"
    ls -la parallel_sim/
else
    echo "Failed to generate waveform file"
fi