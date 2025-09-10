OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
qreg q[5];
creg meas[5];
cy q[2],q[4];
r(0.14079173814132329,2.464666556665951) q[1];
ccz q[0],q[4],q[2];
y q[0];
ecr q[0],q[2];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
