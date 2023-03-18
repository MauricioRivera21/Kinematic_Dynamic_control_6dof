#!/usr/bin/env python

#IN:
#pos articular inicial (q0)
#velocidad articular inicial (dq)
#pos articular deseada (qdes)

#OUT:
#desplazamiento del robot hacia el punto indicado, asi como la aparicion de los markers, uno en el lugar deseado y otro en el efector final del robot

import rospy
from sensor_msgs.msg import JointState
from markers import *
from funciones import *
from roslib import packages

import rbdl

rospy.init_node("control_pdg")
pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
bmarker_actual  = BallMarker(color['RED'])
bmarker_deseado = BallMarker(color['GREEN'])
# Archivos donde se almacenara los datos
fqact = open("/tmp/qactual.dat", "w")
fqdes = open("/tmp/qdeseado.dat", "w")
fxact = open("/tmp/xactual.dat", "w")
fxdes = open("/tmp/xdeseado.dat", "w")

# Nombres de las articulaciones
# Joint names
jnames = ['Corredera6', 'Revolucion5', 'Revolucion4','Revolucion3', 'Revolucion2', 'Revolucion1']
# Objeto (mensaje) de tipo JointState
jstate = JointState()
# Valores del mensaje
jstate.header.stamp = rospy.Time.now()
jstate.name = jnames

# =============================================================
# Configuracion articular inicial (en radianes)
q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Velocidad inicial
dq = np.array([0., 0., 0., 0., 0., 0.])
# Configuracion articular deseada
qdes = np.array([0.5, 2.35619, 0., 0., 0., 0.])
# =============================================================

# Posicion resultante de la configuracion articular deseada
xdes = fkine_ur5(qdes)[0:3,3]

# Copiar la configuracion articular en el mensaje a ser publicado
jstate.position = q
pub.publish(jstate)

# Modelo del robot
modelo = rbdl.loadModel('../urdf/FR_Proyecto_Manipulador.urdf')
#print(dir(modelo))
ndof   = modelo.q_size 	# Grados de libertad


# Frecuencia del envio (en Hz)
freq = 20
dt = 1.0/freq
rate = rospy.Rate(freq)

# Simulador dinamico del robot
robot = Robot(q, dq, ndof, dt)

# Se definen las ganancias del controlador
valores = 1*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Kp = np.diag(valores)
Kd = 2*np.sqrt(Kp)

# Bucle de ejecucion continua
t = 0.0


# Arrays numpy
zeros = np.zeros(ndof)      	# Vector de ceros
tau   = np.zeros(ndof)      	# Para torque
g 	= np.zeros(ndof)      	# Para la gravedad
c 	= np.zeros(ndof)      	# Para el vector de Coriolis+centrifuga
M 	= np.zeros([ndof, ndof])  # Para la matriz de inercia
e 	= np.eye(6)           	# Vector identidad

min_err=0.001


while not rospy.is_shutdown():
    # Leer valores del simulador
    q  = robot.read_joint_positions()
    dq = robot.read_joint_velocities()
    # Posicion actual del efector final
    x = fkine_ur5(q)[0:3,3]
    # Tiempo actual (necesario como indicador para ROS)
    jstate.header.stamp = rospy.Time.now()

    # Almacenamiento de datos
    fxact.write(str(t)+' '+str(x[0])+' '+str(x[1])+' '+str(x[2])+'\n')
    fxdes.write(str(t)+' '+str(xdes[0])+' '+str(xdes[1])+' '+str(xdes[2])+'\n')
    fqact.write(str(t)+' '+str(q[0])+' '+str(q[1])+' '+ str(q[2])+' '+ str(q[3])+' '+str(q[4])+' '+str(q[5])+'\n ')
    fqdes.write(str(t)+' '+str(qdes[0])+' '+str(qdes[1])+' '+ str(qdes[2])+' '+ str(qdes[3])+' '+str(qdes[4])+' '+str(qdes[5])+'\n ')

    # ----------------------------
    # Control dinamico (COMPLETAR)
    # ----------------------------
    u = np.zeros(ndof)   # Reemplazar por la ley de control

    rbdl.InverseDynamics(modelo,q,zeros,zeros,g)

    #torque = Kp.dot(error_q) + Kd.dot(error_dq) # Control PD 
    #torque = Kp.dot(error_q) + Kd.dot(error_dq) + g # Control PD + G
    u = torque
    #print(u)

    # Simulacion del robot
    robot.send_command(u)

    # Publicacion del mensaje
    jstate.position = q
    pub.publish(jstate)
    bmarker_deseado.xyz(xdes)
    bmarker_actual.xyz(x)
    t = t+dt
    # Esperar hasta la siguiente  iteracion
    rate.sleep()

fqact.close()
fqdes.close()
fxact.close()
fxdes.close()