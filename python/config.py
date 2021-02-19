###############################################################################
# config file (python script)
# This code snippet is executed before the command line
# arguments are parsed.
#
# Introduce / use function
#   def config_add_cl_arguments(parser):
#       ...
# to add your own command line options.
#
# Introduce / use
#   def config_after_cl_parsing(args):
#       ...
# to set specific settings according to the parsed command
# line arguments.
###############################################################################

# Use one of 'FATAL','CRITICAL','ERROR','WARNING','WARN','INFO','DEBUG','DEBUG2','DEBUG3','NOTSET' 
LOG_LEVEL = logging.INFO

scanner.SOURCE_DIALECTS = ["cuf","acc"] # one of ["acc","cuf","omp"]
scanner.DESTINATION_DIALECT   = "omp"   # one of ["omp","hip"]

scanner.CONVERT_TO_HIP=[] # list of kernels to convert to HIP. Is ignored if Destination DIALECT is hip.  

scanner.GPUFORT_IFDEF           = "__HIP" 
scanner.CUF_IFDEF           = "CUDA"
# cublas_v1 routines do not have an handle. cublas v2 routines do
scanner.CUBLAS_VERSION      = 1
translator.CUBLAS_VERSION   = scanner.CUBLAS_VERSION
# look for integer (array) with this name; disable: leave empty string or comment out
translator.HINT_CUFFT_PLAN  = r"cufft_plan\w+"
# look for integer (array) with this name; disable: leave empty string or comment out
translator.HINT_CUDA_STREAM = r"stream|strm"

# CUF options
scanner.HIP_MODULE_NAME="hipfort"
scanner.HIP_MATH_MODULE_PREFIX=scanner.HIP_MODULE_NAME+"_"

# ACC options
scanner.ACC_DEV_PREFIX=""
scanner.ACC_DEV_SUFFIX="_d"

def config_add_cl_arguments(parser):
    parser.add_argument('--project',dest="project",type=str,help="Select a project ('dynamico')",required=False)

def config_after_cl_parsing(args):
    # PROJECT SPECIFIC
    if args.project == "dynamico":
        translator.FORTRAN_2_C_TYPE_MAP["complex"]["cstd"] = "hipDoubleComplex"
        translator.FORTRAN_2_C_TYPE_MAP["complex"]["rstd"] = "hipDoubleComplex"
        translator.FORTRAN_2_C_TYPE_MAP["real"]["rstd"] = "double"
        translator.FORTRAN_2_C_TYPE_MAP["complex"]["istd"] = "int"
      
        translator.LOOP_COLLAPSE_STRATEGY="collapse-always"

        fort2hip.FORTRAN_MODULE_PREAMBLE="""
#define rstd 8
        """
        
        global_declarations = """
          integer,parameter::advect_none,advect_vanleer,c4,c8,compress_lim,const,cstd,cyclone,default_nsplit_i,default_nsplit_j,dry_baroclinic,eta_mass,field_t,field_u,field_z,grow_factor,i_2,i4,i_4,i8,i_8,initial_alloc_size,istd,i_std,i_txtslab,k_hadley,k_i,ldown,left,lmdz,lup,max_files,maxlen,maxwritefield,memslabs,method,moist_baroclinic_full,moist_baroclinic_kessler,mountain,nb_face,n_d_fmt,ne_ldown,ne_left,ne_lup,ne_rdown,ne_right,ne_rup,nvert,perturbation,phys_none
          integer,parameter,public::check_basic
          integer,parameter,public::thermo_none
          integer,parameter::r4,r_4,r8,r_8,rdown,right,rstd,rup,size_min,start_unit,supercell,type_integer,type_logical,type_real,vdown,vldown,vlup,vrdown,vrup,vup
          logical,parameter::debug
          logical,parameter::profile_mpi_detail
          real(8),parameter::cly_constant,degrees_to_radians,half_pi,k1_lat_center,k1_lon_center,pi
          real,parameter::grow_factor,one_day,r2es,r3ies,r3les,r4ies,r4les,retv,rtt
          real(rstd),dimension(4),parameter::coef_rk4
          real(rstd),dimension(5),parameter::coef_rk25
          real(rstd),parameter::alpha,daysec,deltap,deltat,epsilon,epsilon0,eta0,etat,gamma,h0,k0,latc,lonc,n,omega0,pb,peq,pi,ps0,pw,q0,qt,r0,rd,rp,t0,tauclee,tau_hadley,teq,ts,u0,u0_hadley,up0,w0_deform,zp,zq1,zq2,zt,zz1      
          INTEGER :: iim,jjm,t_right,t_rup,t_lup,t_ldown,u_right,u_rup,u_lup,u_left,u_ldown,u_rdown,z_rup,z_up,z_lup,z_ldown,z_down,z_rdown
          INTEGER,SAVE :: ll_begin,ll_end,ij_begin,ij_end,ij_begin_ext,ij_end_ext,itau_out,itau_adv,itau_dissip,itau_physics,itaumax,ll_endm1
          REAL(rstd),POINTER :: Ai(:),centroid(:,:),xyz_i(:,:),xyz_e(:,:),xyz_v(:,:),lon_i(:),lon_e(:),lat_i(:),lat_e(:),ep_e(:,:),et_e(:,:),elon_i(:,:),elat_i(:,:),elon_e(:,:),elat_e(:,:),Av(:),de(:),le(:),le_de(:),S1(:,:),S2(:,:),Riv(:,:),Riv2(:,:),Wee(:,:,:),bi(:),fv(:)          
          INTEGER,POINTER    :: ne(:,:)        
          REAL(rstd),SAVE :: radius=6.37122E6
          REAL(rstd),SAVE :: g=9.80616
          REAL(rstd),PARAMETER :: daysec=86400
          REAL(rstd),SAVE :: omega,kappa,cpp,cppv,Rv,Treff,preff,pa,scale_height,scale_factor,gas_constant,mu
          INTEGER,PARAMETER,PUBLIC :: thermo_none,thermo_theta,thermo_entropy,thermo_moist,thermo_boussinesq,thermo_dry,thermo_fake_moist,thermo_moist_debug
          INTEGER, PUBLIC :: caldyn_thermo, physics_thermo
          INTEGER,SAVE :: ll_begin,ll_beginp1,ll_end,ll_endm1,ll_endp1
          INTEGER :: ii_begin,jj_begin,ii_end,jj_end,ii_begin_glo,jj_begin_glo,ii_end_glo,jj_end_glo
          REAL(rstd) :: ptop, dt
          INTEGER :: llm
          INTEGER  :: nqdyn
          INTEGER :: t_rdown, t_left 
          REAL(rstd),POINTER :: adzqw(:,:),dzq(:,:),wq(:,:)   
        """
        scanner.GLOBAL_DECLARATIONS += global_declarations.strip().split("\n")
