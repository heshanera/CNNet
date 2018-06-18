#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU-Linux
CND_DLIB_EXT=so
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/Activation.o \
	${OBJECTDIR}/CNN.o \
	${OBJECTDIR}/ConvolutionLayer.o \
	${OBJECTDIR}/DataProcessor.o \
	${OBJECTDIR}/FCLayer.o \
	${OBJECTDIR}/FileProcessor.o \
	${OBJECTDIR}/PoolLayer.o \
	${OBJECTDIR}/main.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/cnnet

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/cnnet: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/cnnet ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/Activation.o: Activation.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -IEigen -std=c++14 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/Activation.o Activation.cpp

${OBJECTDIR}/CNN.o: CNN.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -IEigen -std=c++14 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/CNN.o CNN.cpp

${OBJECTDIR}/ConvolutionLayer.o: ConvolutionLayer.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -IEigen -std=c++14 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/ConvolutionLayer.o ConvolutionLayer.cpp

${OBJECTDIR}/DataProcessor.o: DataProcessor.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -IEigen -std=c++14 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/DataProcessor.o DataProcessor.cpp

${OBJECTDIR}/FCLayer.o: FCLayer.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -IEigen -std=c++14 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/FCLayer.o FCLayer.cpp

${OBJECTDIR}/FileProcessor.o: FileProcessor.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -IEigen -std=c++14 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/FileProcessor.o FileProcessor.cpp

${OBJECTDIR}/PoolLayer.o: PoolLayer.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -IEigen -std=c++14 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/PoolLayer.o PoolLayer.cpp

${OBJECTDIR}/main.o: main.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -IEigen -std=c++14 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
