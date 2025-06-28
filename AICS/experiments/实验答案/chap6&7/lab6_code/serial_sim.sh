#!/bin/bash
# filepath: /home/HD/course/pe_exp_student/serial_sim.sh

# Create serial_sim directory 
mkdir -p /home/HD/course/pe_exp_student/sim_iverilog/serial_sim

# Enter simulation directory
cd /home/HD/course/pe_exp_student/sim_iverilog

# Compile and run
iverilog -o test -I../src/serial_pe tb_top_0.v ../src/serial_pe/serial_pe.v
./test

# Check if waveform file is generated
if [ -f "serial_sim/serial_pe.vcd" ]; then
    echo "Waveform file generated: serial_sim/serial_pe.vcd"
    ls -la serial_sim/
else
    echo "Failed to generate waveform file"
fi