`timescale 1ns/1ps

module tb_top_0;
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
    // 创建serial_sim目录并dump波形文件
    $dumpfile("serial_sim/serial_pe.vcd");
    $dumpvars(0, tb_top_0);
end
initial begin
    #10000;
    $finish;
end

reg [7  :0] inst[3:0];
reg [511:0] neuron[139:0];
reg [511:0] weight[139:0];
reg [ 44:0] result[3:0];  // 修改位宽以匹配数据生成脚本

initial
begin
  $readmemh("../data/inst", inst);
  $readmemh("../data/neuron", neuron);
  $readmemh("../data/weight", weight);
  $readmemb("../data/result", result);
end

reg [ 1:0]   inst_addr;
reg [12:0]   iter;
reg [15:0]   neuron_addr;
reg [15:0]   weight_addr;

wire [  7:0] pe_inst        = inst[inst_addr];
wire [511:0] pe_weight_line = weight[weight_addr[15:5]];
wire [511:0] pe_neuron_line = neuron[neuron_addr[15:5]];
wire [ 15:0] pe_weight      = pe_weight_line[(16*(31-weight_addr[4:0]))+:16];
wire [ 15:0] pe_neuron      = pe_neuron_line[(16*(31-neuron_addr[4:0]))+:16];
wire [1:0] pe_ctl;
assign pe_ctl[0] = (iter[12:0] == 13'h0);
assign pe_ctl[1] = (iter[12:5] == (pe_inst[7:0] - 1'b1)) && (iter[4:0] == 5'h1f);

reg pe_vld_i;
always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    pe_vld_i <= 1'b0;
  end else if((inst_addr == 2'h0) && (neuron_addr == 16'h0) && (weight_addr == 16'h0)) begin
    pe_vld_i <= 1'b1;
  end else if(pe_ctl[1] && pe_vld_i && (inst_addr == 2'h3)) begin
    pe_vld_i <= 1'b0;
  end
end

//always@(posedge clk or negedge rst_n) begin
//  if(!rst_n) begin
//    iter <= 13'b0;
//  end else if(pe_ctl[1] && pe_vld_i && (inst_addr == 2'h3)) begin
//    iter <= 13'b0;
//  end else if(pe_vld_i) begin
//    iter <= iter + 1'b1;
//  end
//end
always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    iter <= 13'b0;
  end else if(pe_ctl[1] && pe_vld_i) begin  // 每个指令完成后都重置iter
    iter <= 13'b0;
  end else if(pe_vld_i) begin
    iter <= iter + 1'b1;
  end
end


always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    inst_addr <= 2'b0;
  end else if(pe_ctl[1] && pe_vld_i && (inst_addr != 2'h3)) begin
    inst_addr <= inst_addr + 1'b1;
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    weight_addr <= 16'h0;
  end else if(pe_vld_i) begin
    weight_addr <= weight_addr + 1'b1;
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    neuron_addr <= 16'h0;
  end else if(pe_vld_i) begin
    neuron_addr <= neuron_addr + 1'b1;
  end
end

wire [31:0] pe_result;
wire pe_vld_o;
serial_pe u_serial_pe (
  .clk                  (clk      ),
  .rst_n                (rst_n    ),
  .neuron               (pe_neuron),
  .weight               (pe_weight),
  .ctl                  (pe_ctl   ),
  .vld_i                (pe_vld_i ),
  .result               (pe_result),
  .vld_o                (pe_vld_o )
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
  end else if(pe_vld_o && (pe_result != result[result_addr][31:0])) begin  // 只比较低32位
    $display("FAIL: num.%d result not correct!!! Expected: %h, Got: %h", result_addr, result[result_addr][31:0], pe_result);
    compare_pass <= 1'b0;
  end else if(pe_vld_o && (pe_result == result[result_addr][31:0])) begin
    $display("INFO: num.%d result is correct.", result_addr);
  end
end

endmodule
