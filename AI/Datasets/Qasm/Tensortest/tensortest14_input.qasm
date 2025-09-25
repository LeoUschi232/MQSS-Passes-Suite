OPENQASM 2.0;
include "qelib1.inc";
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
qreg q[5];
creg meas[5];
crz(-1.5556787644530061) q[1],q[3];
csdg q[1],q[3];
rzx(1.6976012350924776) q[2],q[4];
crx(2.1324993748550787) q[4],q[2];
ryy(-2.563030269930782) q[1],q[4];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
