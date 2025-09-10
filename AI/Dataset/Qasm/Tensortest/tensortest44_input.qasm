OPENQASM 2.0;
include "qelib1.inc";
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
qreg q[5];
creg meas[5];
ecr q[3],q[4];
cry(2.7527358025348976) q[3],q[4];
rzz(-1.013036684267274) q[3],q[0];
x q[1];
cz q[4],q[1];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
