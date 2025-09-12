OPENQASM 2.0;
include "qelib1.inc";
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
qreg q[5];
creg meas[5];
cry(-2.9562918784808834) q[2],q[4];
ccz q[1],q[0],q[4];
cz q[0],q[3];
rz(2.3418773480826847) q[1];
ccx q[2],q[0],q[4];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
