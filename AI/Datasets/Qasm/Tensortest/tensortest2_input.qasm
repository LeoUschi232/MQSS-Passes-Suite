OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
qreg q[5];
creg meas[5];
rz(-3.0691750825340574) q[0];
r(0.5471328721505899,1.648513922159296) q[1];
r(-0.5344808960299678,-2.4853158670745494) q[3];
tdg q[3];
ecr q[2],q[1];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
