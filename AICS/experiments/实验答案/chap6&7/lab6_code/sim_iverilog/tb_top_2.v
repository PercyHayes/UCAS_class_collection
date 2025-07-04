`timescale 1ns/1ps

module tb_top_2;
localparam WIDTH = 32;

reg        clk; // clk
reg        rst_n; // reset

initial begin
    clk = 1'b0;
    forever #0.5 clk = ~clk;
end
initial begin
    rst_n = 1'b0;
    #10 rst_n = 1'b1;
end
initial begin
    $dumpfile("matrix_sim/matrix_pe.vcd");
    $dumpvars(0, tb_top_2);
end
initial begin
    #10000;
    $finish;
end

reg [7  :0] inst[3:0];
reg [511:0] neuron[139:0];
reg [511:0] weight[139:0];
reg [ 44:0] result[3:0];//要改位宽

initial
begin
  // path
  $readmemh("../data/inst", inst);
  $readmemh("../data/neuron", neuron);
  $readmemh("../data/weight", weight);
  $readmemb("../data/result", result);
end

reg [ 1:0]   inst_addr;
reg [ 7:0]   iter;
reg [15:0]   neuron_addr;
reg [15:0]   weight_addr;

wire [  7:0] pe_inst   = inst[inst_addr];
wire [511:0] pe_weight = weight[weight_addr];
wire [511:0] pe_neuron = neuron[neuron_addr];

reg          ib_ctl_uop_valid;
wire         ib_ctl_uop_ready;
reg          wram_mpe_weight_valid;
wire         wram_mpe_weight_ready;
reg          nram_mpe_neuron_valid;
wire         nram_mpe_neuron_ready;


// 加种子
integer seed = 123;

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    inst_addr <= 2'b0;
  end else if(ib_ctl_uop_valid && ib_ctl_uop_ready && (inst_addr != 2'h3)) begin
    inst_addr <= inst_addr + 1'b1;
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    weight_addr <= 16'h0;
  end else if(wram_mpe_weight_valid && wram_mpe_weight_ready) begin
    weight_addr <= weight_addr + 1'b1;
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    neuron_addr <= 16'h0;
  end else if(nram_mpe_neuron_valid && nram_mpe_neuron_ready) begin
    neuron_addr <= neuron_addr + 1'b1;
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    ib_ctl_uop_valid <= 1'b0;
  end else if(ib_ctl_uop_valid && ib_ctl_uop_ready) begin
    ib_ctl_uop_valid <= ($random(seed) % 2);
  end else if(!ib_ctl_uop_valid) begin
    ib_ctl_uop_valid <= ($random(seed) % 2);
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    wram_mpe_weight_valid <= 1'b0;
  end else if(wram_mpe_weight_valid && wram_mpe_weight_ready) begin
    wram_mpe_weight_valid <= ($random(seed) % 2);
  end else if(!wram_mpe_weight_valid) begin
    wram_mpe_weight_valid <= ($random(seed) % 2);
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    nram_mpe_neuron_valid <= 1'b0;
  end else if(nram_mpe_neuron_valid && nram_mpe_neuron_ready) begin
    nram_mpe_neuron_valid <= ($random(seed) % 2);
  end else if(!nram_mpe_neuron_valid) begin
    nram_mpe_neuron_valid <= ($random(seed) % 2);
  end
end

wire [31:0] pe_result;
wire pe_vld_o;
matrix_pe u_matrix_pe (
  .clk                  (clk                  ),
  .rst_n                (rst_n                ),

  .nram_mpe_neuron      (pe_neuron            ),
  .nram_mpe_neuron_valid(nram_mpe_neuron_valid),
  .nram_mpe_neuron_ready(nram_mpe_neuron_ready),

  .wram_mpe_weight      (pe_weight            ),
  .wram_mpe_weight_valid(wram_mpe_weight_valid),
  .wram_mpe_weight_ready(wram_mpe_weight_ready),

  .ib_ctl_uop           (pe_inst              ),
  .ib_ctl_uop_valid     (ib_ctl_uop_valid     ),
  .ib_ctl_uop_ready     (ib_ctl_uop_ready     ),

  .result               (pe_result            ),
  .vld_o                (pe_vld_o             )
);

reg [1:0] result_addr;
reg compare_pass;
always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    result_addr <= 2'h0;
  end else if(pe_vld_o) begin
    result_addr <= result_addr + 1'b1;
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    compare_pass <= 1'b1;
  end else if(pe_vld_o && (pe_result != result[result_addr][31:0])) begin
    $display("FAIL: num.%d result not correct!!!", result_addr);
    compare_pass <= 1'b0;
  end else if(pe_vld_o && (pe_result == result[result_addr][31:0])) begin
    $display("INFO: num.%d result is correct.", result_addr);
  end
end

endmodule
