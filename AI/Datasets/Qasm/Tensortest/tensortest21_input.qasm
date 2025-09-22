OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
qreg q[5];
creg meas[5];
sdg q[3];
r(-2.9441012407579406,-2.283386678727042) q[1];
t q[3];
rz(2.926948982413487) q[1];
iswap q[0],q[2];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
