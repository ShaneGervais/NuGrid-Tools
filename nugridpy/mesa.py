#
# NuGridpy - Tools for accessing and visualising NuGrid data.
#
# Copyright 2007 - 2014 by the NuGrid Team.
# All rights reserved. See LICENSE.
#

"""
MESA output data loading and plotting

v0.2, 15OCT2012: NuGrid collaboration
(Sam Jones, Michael Bennett, Daniel Conti, William Hillary,
Falk Herwig, Christian Ritter)

v0.1, 23JUN2010: Falk Herwig

    mesa.py provides tools to get MESA stellar evolution data output
    into your favourite python session.  In the LOGS directory MESA
    outputs two types of files: history.data or star.log is a time
    evolution output, printing one line per so many cycles (e.g. each
    cycle) of all sorts of things.  profilennn.data or lognnn.data
    files are profile data files.  nnn is the number of profile.data
    or log.data files that is translated into model cycles in the
    profiles.index file.

    MESA allows users to freely define what should go into these two
    types of outputs, which means that column numbers can and do
    change.  mesa.py reads in both types of files and present them (as
    well as any header attributes) as arrays that can be referenced by
    the actual column name as defined in the header section of the
    files.  mesa.py then defines a (hopefully growing) set of standard
    plots that make use of the data just obtained.

    mesa.py is organised as a module that can be imported into any
    python or ipython session. It is related to nugridse.py which is a
    similar module to deal with 'se' output, used by the NuGrid
    collaboration.  mesa.py does not need se libraries.  The 'se' output
    files that can be written with MESA can be read and processed with
    the nugridse.py tool.

    mesa.py is providing two class objects, mesa_profile and
    history_data.  The first makes profile data available, the second
    reads and plots the history.data or star.log file.  Note that
    several instances of these can be initiated within one session and
    data from different instances (i.e. models, tracks etc) can be
    overplotted.

    Here is how a simple session could look like that is plotting an
    HRD (We prefer to load ipython with matplotlib and numpy support
    via the alias:

    alias mpython='ipython --pylab')

    >>> import mesa as ms
    >>> help ms
    ------> help(ms)

    >>> s=ms.history_data('.')
    >>> s.hrd()

    In order to find out what header attributes and columns are
    available in history.data or star.log use:

    >>> s.header_attr
    {'burn_min1': 50.0,
     'burn_min2': 1000.0,
     'c12_boundary_limit': 0.0001,
     'h1_boundary_limit': 0.0001,
     'he4_boundary_limit': 0.0001,
     'initial_mass': 2.0,
     'initial_z': 0.01}

    >>> s.cols
    {'center_c12': 38,
     'center_h1': 36,
     'center_he4': 37,
     ...

    In order to read the profile data from the first profile.data file
    in profiles.index, and then get the mass and temperature out and
    finally plot them try.  Typically you will have already a
    Kippenhahn diagram as a function of model number in front of you,
    and you want to access profile information for a given cycle
    number.  Typically you do not have profiles for all cycle
    numbers.  The best way to start a profile instance is with
    num_type='nearest_model' (check the docstring for other ways to
    select profiles for a profile instance):

    >>> a1=ms.mesa_profile('LOGS',59070)
    2001 in profiles.index file ...
    Found and load nearest profile for cycle 59000
    reading LOGS/profile1801.data ...
    Closing profile tool ...

    >>> T=a1.get('temperature')
    >>> mass=a1.get('mmid')
    >>> plot(mass,T)
    [<matplotlib.lines.Line2D object at 0x8456ed0>]

    Or, you could have had it easier in the following way:

    >>> a1.plot('mass','c12',logy=True,shape='-',legend='$^{12}\mathrm{C}$')

    where the superclass plot method interprets data column headers
    correctly and does all the work for you.

    Of course, a1.cols etc are available here as well and many other
    things. E.g. a.model contains an array with all the models for
    which profile.data or log.data  are available. You may initiate a profile object
    with a model number:

    >>> a2=ms.mesa_profile('.',55000,num_type='model')
    100 in profiles.index file ...
    reading ./profile87.data ...

    a1.log_ind (for any profile instance) provides a map of model
    number to profile file number.
    a1.cols and a1.header_attr gives the column names and header attributes.

"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
from numpy import *
from math import *
import numpy as np
import numpy as np
import matplotlib
import matplotlib.pylab as pyl
import matplotlib.pyplot as pl
from matplotlib.collections import Collection
from matplotlib.artist import allow_rasterization
from matplotlib.patches import PathPatch
import os
import sys

from . import astronomy as ast
from . import ascii_table
from . import utils as u
from .plot_common import PlotCommon
from .mesa_plots.plotting import MesaPlotMixin


def set_nugrid_path(path):
    """
        This function sets the path to the NuGrid VOSpace directory as a
        global variable, so that it need only be set once during an inter-
        active session.
        """
    global nugrid_path
    nugrid_path=path

def set_nice_params():
    fsize=18

    params = {'axes.labelsize':  fsize,
    #    'font.family':       'serif',
    'font.family':        'Times New Roman',
    'figure.facecolor':  'white',
    'text.fontsize':     fsize,
    'legend.fontsize':   fsize,
    'xtick.labelsize':   fsize*0.8,
    'ytick.labelsize':   fsize*0.8,
    'ytick.minor.pad': 8,
    'ytick.major.pad': 8,
    'xtick.minor.pad': 8,
    'xtick.major.pad': 8,
    'text.usetex':       False,
    'lines.markeredgewidth': 0}
    pl.rcParams.update(params)

class mesa_profile(PlotCommon):
    """
    read profiles.index and prepare reading MESA profile files

    starts with reading profiles.index and creates hash array
    profile.data can then be accessed via prof_plot

    Parameters
    ----------
    sldir : string
        Directory path of LOGS.
    num : integer
        by default this is the i.  profile file (profile.data or
        log.data) available (e.g. num=1 is the 1. available profile
        file), however if you give
    num_type : string, optional
        If 'model' (exact) or 'nearest_model': get the profile
        profile.data file for model (or cycle number) used by the
        stellar evolution code

        If 'profile_num': num will be interpreted as the
        profile.data or log.data number profile_num (profile_num is
        the number that appears in the file names of type
        profile23.data or log23.data)

        If 'profiles_i': the ith file in profiles.index file

        If 'explicit': the exact file path of the profile file
        should be given in the variable give_filename.

        The default is "nearest_model".
    prof_ind_name : string, optional
        Use this optional argument if the profiles.index file has
        an alternative name, for example, do
        superpro=ms.profile('LOGS',1,prof_ind_name='super.prof')
    profile_prefix : string, optional
        Prifix in the profile name.  The default is "profile".
    data_suffix : string, optional
        Optional arguments that allow you to change the defaults
        for the profile.data profile files.
    mass : integer or float, optional
        The user may select a mass and metallicity instead of providing
        the sldir explicitly, if they are using the VOSpace data. If mass
        is provided then Z should also be provided.
        The default is None (i.e. user gives sldir explicitly)
    Z : float, optional
        See 'mass' above.
        The default is None (i.e. user gives sldir explicitly)
    data_set : string, optional
        Coose your data  of 'set1' or 'set1ext'.  The default is 'set1ext'.


    Examples
    --------
    initialise a mesa_profile instance for cycle 2000 like this:

    >>>my_profile1=ms.mesa_profile('LOGS',2000)

    or like this:

    >>>my_profile2=ms.mesa_profile(mass=2,Z=0.01,num=2000)
    """

    sldir = ''

    def __init__(self, sldir='./LOGS', num=1, num_type='nearest_model',
                 prof_ind_name='profiles.index',
                 profile_prefix='profile', data_suffix='.data', mass=None,
                 Z=None,give_filename=None,data_set='set1ext'):
        """
        read a profile.data profile file

        Parameters
        ----------
        sldir : string
            Directory path of LOGS.
        num : integer
            by default this is the i.  profile file (profile.data or
            log.data) available (e.g. num=1 is the 1. available profile
            file), however if you give
        num_type : string, optional
            If 'model' (exact) or 'nearest_model': get the profile
            profile.data file for model (or cycle number) used by the
            stellar evolution code

            If 'profile_num': num will be interpreted as the
            profile.data or log.data number profile_num (profile_num is
            the number that appears in the file names of type
            profile23.data or log23.data)

            If 'profiles_i': the ith file in profiles.index file

            If 'explicit': the exact file path of the profile file
            should be given in the variable give_filename.

            The default is "nearest_model".
        prof_ind_name : string, optional
            Use this optional argument if the profiles.index file has
            an alternative name, for example, do
            superpro=ms.profile('LOGS',1,prof_ind_name='super.prof')
        profile_prefix : string, optional
            Prifix in the profile name.  The default is "profile".
        data_suffix : string, optional
            Optional arguments that allow you to change the defaults
            for the profile.data profile files.
        mass : integer or float, optional
            The user may select a mass and metallicity instead of providing
            the sldir explicitly, if they are using the VOSpace data. If mass
            is provided then Z should also be provided.
            The default is None (i.e. user gives sldir explicitly)
        Z : float, optional
            See 'mass' above.
            The default is None (i.e. user gives sldir explicitly)

        Examples
        --------
        initialise a mesa_profile instance for cycle 2000 like this:

        >>>my_profile1=ms.mesa_profile('LOGS',2000)

        or like this:

        >>>my_profile2=ms.mesa_profile(mass=2,Z=0.01,num=2000)

        """

        self.prof_ind_name = prof_ind_name
        self.sldir         = sldir

        # seeker to find the data requested on VOspace:
        if mass is not None and Z is not None:
            try:
                print('nugrid_path = '+nugrid_path)
            except:
                raise IOError("nugrid_path has not been set. This is the path to the NuGrid VOSpace, e.g. /tmp/NuGrid. Set this using mesa.set_nugrid_path('path')")

            # which set? [find nearest]

            if (data_set=='set1ext'):
               setsZs=[0.02,0.01,6.e-3,1.e-3,1.e-4]
               setsnames=['set1.2','set1.1','set1.3a','set1.4a','set1.5a']
            elif (data_set=='set1'):
               setsZs=[0.02,0.01]
               setsnames=['set1.2','set1.1']
            else:
               raise IOError("Sorry. Requested data_set not available. Choose between set1ext and set1.")                  
       
            idx=np.abs(np.array(setsZs)-Z).argmin()
            setname=setsnames[idx]
            realZ=setsZs[idx]

            print('closest set is '+setname+' (Z = '+str(realZ)+')')

            mod_dir = nugrid_path+'/data/'+data_set+'/'+setname+'/see_wind/'
            if not os.path.exists(mod_dir):
                print('mod_dir = ', mod_dir)
                raise IOError("The data does not seem to be here. Please check that the NuGrid VOSpace is mounted and nugrid_path has been set correctly using mesa.set_nugrid_path('path')'.")

            # which mass? [find nearest]
            list=[el for el in os.listdir(mod_dir) if el[0]=='M']
            if len(list) == 0:
                raise IOError("Sorry. There is no data available for this set at present: "+mod_dir)
            setmasses=[el[1:el.index('Z')] for el in list]
            for i in range(len(setmasses)):
                if setmasses[i][-1]=='.': setmasses[i]=setmasses[i][:-1]
                setmasses[i] = float(setmasses[i])
            idx2=np.abs(np.array(setmasses)-mass).argmin()
            modname=list[idx2]
            realmass=setmasses[idx2]

            print('closest mass is '+str(realmass))

            mod_dir+=modname
            if 'LOGS' not in os.listdir(mod_dir):
                raise IOError("No 'LOGS' directory for this model. It may have been computed with the Geneva code. Try nugridse.py to explore the see_wind data for this model.")
            else:
                self.sldir=mod_dir+'/LOGS'
                sldir = mod_dir+'/LOGS'

        if give_filename is not None and num_type is not 'explicit':
            raise KeyError("Exact filename given but num_type is not explicit.")

        if num_type is 'nearest_model' or num_type is 'model':
            self._profiles_index()
        if num_type is 'nearest_model':
            amods=array(self.model)
            if amods[0]>num:
                 num = amods[0]  
            elif amods[-1]<num:
                 num = amods[-1]
            else:
                 nearmods=[where(amods<=num)[0][-1],where(amods>=num)[0][0]]
                 sometable={}
                 for thing in nearmods:
                     sometable[abs(self.model[thing]-num)]=thing
                 nearest = min(abs(self.model[nearmods[0]]-num),\
                   abs(self.model[nearmods[1]]-num))
                 num = self.model[sometable[nearest]]
            print('Found and load nearest profile for cycle '+str(num))
            num_type = 'model'
        if num_type is 'model':
            try:
                log_num=self.log_ind[num]
            except KeyError:
                print('There is no profile file for this model')
                print("You may retry with num_type='nearest_model'")
                return
        elif num_type is 'profiles_i':
            log_num=self._log_file_ind(num)
            if log_num == -1:
                print("Could not find a profile file with that number")
                return
        elif num_type is 'profile_num':
            log_num = num
        elif num_type is 'explicit':
            pass
        else:
            print('unknown num_type')
            return

        if num_type is 'explicit':
            filename = sldir+'/'+give_filename
            if not os.path.exists(filename):
                print('error: file '+give_filename+' not found in '+sldir)
        else:
            filename=self.sldir+'/'+profile_prefix+str(log_num)+data_suffix
            if not os.path.exists(filename):
                profile_prefix='log'
                filename=self.sldir+'/'+profile_prefix+str(log_num)+data_suffix
                if not os.path.exists(filename):
                    print('error: no profile.data file found in '+sldir)
                    print('error: no log.data file found in '+sldir)


        print('reading profile'+filename+' ...')
        header_attr = _read_mesafile(filename,only='header_attr')
        num_zones=int(header_attr['num_zones'])
        header_attr,cols,data = _read_mesafile(filename,data_rows=num_zones,only='all')

        self.cols        = cols
        self.header_attr = header_attr
        self.data        = data


    def __del__(self):
        print('Closing profile tool ...')

    def _profiles_index(self):
        """
        read profiles.index and make hash array

        Notes
        -----
        sets the attributes.

        log_ind : hash array that returns profile.data or log.data
        file number from model number.

        model : the models for which profile.data or log.data is
        available

        """

        prof_ind_name = self.prof_ind_name

        f = open(self.sldir+'/'+prof_ind_name,'r')
        line = f.readline()
        numlines=int(line.split()[0])
        print(str(numlines)+' in profiles.index file ...')

        model=[]
        log_file_num=[]
        for line in f:
            model.append(int(line.split()[0]))
            log_file_num.append(int(line.split()[2]))

        log_ind={}    # profile.data number from model
        for a,b in zip(model,log_file_num):
            log_ind[a] = b

        self.log_ind=log_ind
        self.model=model

# let's start with functions that aquire data

    def _log_file_ind(self,inum):
        """
        Information about available profile.data or log.data files.

        Parameters
        ----------
        inum : integer
            Attempt to get number of inum's profile.data file.
            inum_max: max number of profile.data or log.data files
            available

        """

        self._profiles_index()
        if inum <= 0:
            print("Smallest argument is 1")
            return

        inum_max = len(self.log_ind)
        inum -= 1

        if inum > inum_max:
            print('There are only '+str(inum_max)+' profile file available.')
            log_data_number = -1
            return log_data_number
        else:
            log_data_number=self.log_ind[self.model[inum]]
            print('The '+str(inum+1)+'. profile.data file is '+ \
                  str(log_data_number))
            return log_data_number

    def get(self,str_name):
        """
        return a column of data with the name str_name.

        Parameters
        ----------
        str_name : string
            Is the name of the column as printed in the
            profilennn.data or lognnn.data file; get the available
            columns from self.cols (where you replace self with the
            name of your instance)

        """

        column_array = self.data[:,self.cols[str_name]-1].astype('float')
        return column_array

    def write_PROM_HOTB_progenitor(self,name,description):
        """
        Write a progenitor file for the PROMETHEUS/HBOT supernova code.

        Parameters
        ----------
        name : string
            File name for the progenitor file
        description : string
            Information to be written into the file header.
        """
        try:
            from ProgenitorHotb_new import ProgenitorHotb_new
        except ImportError:
            print('Module ProgenitorHotb_new not found.')
            return
        
        nz=len(self.get('mass'))
        prog=ProgenitorHotb_new(nz)

        prog.header = '#'+description+'\n'

        prog.xzn    = self.get('rmid')[::-1]*ast.rsun_cm
        prog.massb  = self.get('mass')[::-1]
        prog.r_ob   = max(self.get('radius'))*ast.rsun_cm
        prog.temp   = 10.**self.get('logT')[::-1]*8.620689655172413e-11 # in MeV
        prog.stot   = self.get('entropy')[::-1]
        prog.ye     = self.get('ye')[::-1]
        prog.densty = 10.**self.get('logRho')[::-1]
        prog.press  = 10.**self.get('logP')[::-1]
        prog.eint   = self.get('energy')[::-1]
        prog.velx   = self.get('velocity')[::-1]

        nuclei=['neut','h1','he4','c12','o16','ne20','mg24','si28','s32',
                'ar36','ca40','ti44','cr48','fe52','fake']

        for i in range(len(nuclei)):
            if nuclei[i] == 'fake':
                ni56 = self.get('fe56')+self.get('cr56')
                prog.xnuc[:,i] = ni56[::-1]
            else:
                prog.xnuc[:,i] = self.get(nuclei[i])[::-1]

        prog.write(name)

    def write_STELLA_model(self,name):
        """
        Write an initial model in a format that may easily be read by the
        radiation hydrodynamics code STELLA.

        Parameters
        ----------
        name : string
            an identifier for the model. There are two output files from
            this method, which will be <name>.hyd and <name>.abn, which
            contain the profiles for the hydro and abundance variables,
            respectively.

        """

        # Hydro variables:
        zn = np.array(self.get('zone'),np.int64)
        Mr = self.get('mass')[::-1]
        dM = 10. ** self.get('logdq')[::-1] * self.header_attr['star_mass']
        R = self.get('radius')[::-1] * ast.rsun_cm
        dR = np.insert( np.diff(R), 0, R[0] )
        Rho = 10. ** self.get('logRho')[::-1]
        PRE = 10. ** self.get('logP')[::-1]
        T = 10. ** self.get('logT')[::-1]
        V = self.get('velocity')[::-1]

        # Abundances:
        def make_list(element,lowA,highA):
            l = []
            for i in range(lowA,highA+1):
                l.append(element+str(i))
            return l

        abun_avail = list(self.cols.keys())

        def elemental_abund(ilist,abun_avail):
            X = np.zeros(len(self.get('mass')))
            for a in ilist:
                if a in abun_avail:
                    X += self.get(a)[::-1]

            return X

        iH = ['h1','h2','prot']
        XH  = elemental_abund(iH, abun_avail)
        XHe = elemental_abund(make_list('he',1,5), abun_avail)
        XC  = elemental_abund(make_list('c',11,15), abun_avail)
        XN  = elemental_abund(make_list('n',12,16), abun_avail)
        XO  = elemental_abund(make_list('o',13,20), abun_avail)
        XNe = elemental_abund(make_list('ne',17,25), abun_avail)
        XNa = elemental_abund(make_list('na',20,25), abun_avail)
        XMg = elemental_abund(make_list('mg',21,28), abun_avail)
        XAl = elemental_abund(make_list('al',21,30), abun_avail)
        XSi = elemental_abund(make_list('si',25,34), abun_avail)
        XS  = elemental_abund(make_list('s',28,38), abun_avail)
        XAr = elemental_abund(make_list('ar',32,46), abun_avail)
        XCa = elemental_abund(make_list('ca',36,53), abun_avail)
        XFe = elemental_abund(make_list('fe',50,65), abun_avail)
        XCo = elemental_abund(make_list('co',52,66), abun_avail)
        XNi = elemental_abund(make_list('ni',54,71), abun_avail)

        XNi56 = self.get('ni56')

        # Write the output files:
        file_hyd = name+'.hyd'
        file_abn = name+'.abn'

        f = open(file_hyd,'w')
        # write header:
        f.write('  0.000E+00\n')
        f.write('# No.')
        f.write('Mr'.rjust(28)+
                'dM'.rjust(28)+
                'R'.rjust(28)+
                'dR'.rjust(28)+
                'Rho'.rjust(28)+
                'PRE'.rjust(28)+
                'T'.rjust(28)+
                'V'.rjust(28)+
                '\n')
        # write data:
        for i in range(len(zn)):
            f.write( str(zn[i]).rjust(5) +
                    '%.16E'.rjust(11) %Mr[i] +
                    '%.16E'.rjust(11) %dM[i] +
                    '%.16E'.rjust(11) %R[i] +
                    '%.16E'.rjust(11) %dR[i] +
                    '%.16E'.rjust(11) %Rho[i] +
                    '%.16E'.rjust(11) %PRE[i] +
                    '%.16E'.rjust(11) %T[i] +
                    '%.16E'.rjust(11) %V[i] +
                    '\n')

        f.close()

        f = open(file_abn,'w')
        # write header:
        f.write('# No.')
        f.write('Mr'.rjust(28)+
                'H'.rjust(28)+
                'He'.rjust(28)+
                'C'.rjust(28)+
                'N'.rjust(28)+
                'O'.rjust(28)+
                'Ne'.rjust(28)+
                'Na'.rjust(28)+
                'Mg'.rjust(28)+
                'Al'.rjust(28)+
                'Si'.rjust(28)+
                'S'.rjust(28)+
                'Ar'.rjust(28)+
                'Ca'.rjust(28)+
                'Fe'.rjust(28)+
                'Co'.rjust(28)+
                'Ni'.rjust(28)+
                'X(56Ni)'.rjust(28)+
                '\n')
        # write data:
        for i in range(len(zn)):
            f.write( str(zn[i]).rjust(5) +
                    '%.16E'.rjust(11) %Mr[i] +
                    '%.16E'.rjust(11) %XH[i] +
                    '%.16E'.rjust(11) %XHe[i] +
                    '%.16E'.rjust(11) %XC[i] +
                    '%.16E'.rjust(11) %XN[i] +
                    '%.16E'.rjust(11) %XO[i] +
                    '%.16E'.rjust(11) %XNe[i] +
                    '%.16E'.rjust(11) %XNa[i] +
                    '%.16E'.rjust(11) %XMg[i] +
                    '%.16E'.rjust(11) %XAl[i] +
                    '%.16E'.rjust(11) %XSi[i] +
                    '%.16E'.rjust(11) %XS[i] +
                    '%.16E'.rjust(11) %XAr[i] +
                    '%.16E'.rjust(11) %XCa[i] +
                    '%.16E'.rjust(11) %XFe[i] +
                    '%.16E'.rjust(11) %XCo[i] +
                    '%.16E'.rjust(11) %XNi[i] +
                    '%.16E'.rjust(11) %XNi56[i] +
                    '\n')

    def write_LEAFS_model(self,nzn=30000000,dr=5.e4,
                          rhostrip=5.e-4):
        """
        write an ascii file that will be read by Sam's version of
        inimod.F90 in order to make an initial model for LEAFS
        """

        from scipy import interpolate

        ye = self.get('ye')
        newye=[]
        rho = 10.**self.get('logRho')[::-1] # centre to surface
        # get index to strip all but the core:
        idx = np.abs(rho - rhostrip).argmin() + 1
        rho = rho[:idx]
        rhoc = rho[0]
        rad  = 10.**self.get('logR') * ast.rsun_cm
        rad = rad[::-1][:idx]
        ye = ye[::-1][:idx]

        print('there will be about ',old_div(rad[-1], dr), 'mass cells...')

        # add r = 0 point to all arrays
        rad = np.insert(rad,0,0)
        ye  = np.insert(ye,0,ye[0])
        rho = np.insert(rho,0,rho[0])

        print(rad)

        # interpolate
        fye  = interpolate.interp1d(rad,ye)
        frho = interpolate.interp1d(rad,rho)
        newye = []
        newrho = []
        newrad = []

        Tc   = 10.**self.get('logT')[-1]

        for i in range(nzn):
            if i * dr > rad[-1]: break
            newye.append(fye( i * dr ))
            newrho.append(frho( i * dr ))
            newrad.append( i * dr )

        f = open('M875.inimod','w')

        f.write(str(Tc)+' \n')
        f.write(str(rhoc)+' \n')
        for i in range(len(newye)):
            f.write(str(i+1)+'  '+str(newrad[i])+'  '+\
                    str(newrho[i])+'  '+str(newye[i])+' \n')

        f.close()

    def energy_profile(self,ixaxis):
        """
            Plot radial profile of key energy generations eps_nuc,
            eps_neu etc.

            Parameters
            ----------
            ixaxis : 'mass' or 'radius'
        """

        mass = self.get('mass')
        radius = self.get('radius') * ast.rsun_cm
        eps_nuc = self.get('eps_nuc')
        eps_neu = self.get('non_nuc_neu')

        if ixaxis == 'mass':
            xaxis = mass
            xlab = 'Mass / M$_\odot$'
        else:
            xaxis = old_div(radius, 1.e8) # Mm
            xlab = 'Radius (Mm)'

        pl.plot(xaxis, np.log10(eps_nuc),
                'k-',
                label='$\epsilon_\mathrm{nuc}>0$')
        pl.plot(xaxis, np.log10(-eps_nuc),
                'k--',
                label='$\epsilon_\mathrm{nuc}<0$')
        pl.plot(xaxis, np.log10(eps_neu),
                'r-',
                label='$\epsilon_\\nu$')

        pl.xlabel(xlab)
        pl.ylabel('$\log(\epsilon_\mathrm{nuc},\epsilon_\\nu)$')
        pl.legend(loc='best').draw_frame(False)

class history_data(MesaPlotMixin):
    """
    read history.data or star.log MESA output and plot various things,
    including HRD, Kippenhahn etc

    Parameters
    ----------
    sldir : string
        which LOGS directory.
    slname : string, optional
        If star.log is available instead, star.log file is read, this
        is an optional argument if history.data or star.log file has
        an alternative name.  The default is "history.data".
    clean_starlog : boolean, optional
        Request new cleaning of history.data or star.log, makes
        history.datasa or star.logsa which is the file that is actually
        read and plotted.  The default is False.
    mass : integer or float, optional
        The user may select a mass and metallicity instead of providing
        the sldir explicitly, if they are using the VOSpace data. If mass
        is provided then Z should also be provided.
        The default is None (i.e. user gives sldir explicitly)
    Z : float, optional
        See 'mass' above.
        The default is None (i.e. user gives sldir explicitly)
    data_set : string, optional
        Coose your data  of 'set1' or 'set1ext'.  The default is 'set1ext'.

    Examples
    --------
    use like this:

    >>> another=ms.history_data('LOGS',slname='anothername')

    or this:

    >>> ms.set_nugrid_path('/tmp/NuGrid')
    >>> anotherone=ms.history_data(mass=2,Z=0.01)
    """

    sldir  = ''
    slname = ''
    header_attr = []
    cols = []

    def __init__(self, sldir='./LOGS', slname='history.data',
                 clean_starlog=False, mass=None, Z=None,data_set='set1ext'):
        self.sldir  = sldir
        self.slname = slname
        self.clean_starlog  = clean_starlog

        # seeker to find the data requested on VOspace:
        if mass is not None and Z is not None:
            try:
                print('nugrid_path = '+nugrid_path)
            except:
                raise IOError("nugrid_path has not been set. This is the path to the NuGrid VOSpace, e.g. /tmp/NuGrid. Set this using mesa.set_nugrid_path('path')")

            # which set? [find nearest]

            if (data_set=='set1ext'):
               setsZs=[0.02,0.01,6.e-3,1.e-3,1.e-4]
               setsnames=['set1.2','set1.1','set1.3a','set1.4a','set1.5a']
            elif (data_set=='set1'):
               setsZs=[0.02,0.01]
               setsnames=['set1.2','set1.1']
            else:
               raise IOError("Sorry. Requested data_set not available. Choose between set1ext and set1.")                  

            idx=np.abs(np.array(setsZs)-Z).argmin()
            setname=setsnames[idx]
            realZ=setsZs[idx]

            print('closest set is '+setname+' (Z = '+str(realZ)+')')

            mod_dir = nugrid_path+'/data/'+data_set+'/'+setname+'/see_wind/'
            if not os.path.exists(mod_dir):
                print('mod_dir = ', mod_dir)
                raise IOError("The data does not seem to be here. Please check that the NuGrid VOSpace is mounted and nugrid_path has been set correctly using mesa.set_nugrid_path('path')'.")

            # which mass? [find nearest]
            list=[el for el in os.listdir(mod_dir) if el[0]=='M']
            if len(list) == 0:
                raise IOError("Sorry. There is no data available for this set at present: "+mod_dir)

            setmasses=[el[1:el.index('Z')] for el in list]
            for i in range(len(setmasses)):
                if setmasses[i][-1]=='.': setmasses[i]=setmasses[i][:-1]
                setmasses[i] = float(setmasses[i])
            idx2=np.abs(np.array(setmasses)-mass).argmin()
            modname=list[idx2]
            realmass=setmasses[idx2]

            print('closest mass is '+str(realmass))

            mod_dir+=modname
            if 'LOGS' not in os.listdir(mod_dir):
                raise IOError("No 'LOGS' directory for this model. It may have been computed with the Geneva code. Try nugridse.py to explore the see_wind data for this model.")
            else:
                self.sldir=mod_dir+'/LOGS'
                sldir=mod_dir+'/LOGS'

        if not os.path.exists(self.sldir+'/'+self.slname):
            if not os.path.exists(self.sldir+'/'+'star.log'):
                print('error: no history.data file found in '+sldir)
                print('error: no star.log file found in '+sldir)
            else:
                self.slname='star.log'
                self._read_starlog()
        else:
            self._read_starlog()

    def __del__(self):
        print('Closing', self.slname,' tool ...')

# let's start with functions that aquire data
    def _read_starlog(self):
        """ read history.data or star.log file again"""

        sldir   = self.sldir
        slname  = self.slname
        slaname = slname+'sa'

        if not os.path.exists(sldir+'/'+slaname):
            print('No '+self.slname+'sa file found, create new one from '+self.slname)
            _cleanstarlog(sldir+'/'+slname)
        else:
            if self.clean_starlog:
                print('Requested new '+self.slname+'sa; create new from '+self.slname)
                _cleanstarlog(sldir+'/'+slname)
            else:
                print('Using old '+self.slname+'sa file ...')

        cmd=os.popen('wc '+sldir+'/'+slaname)
        cmd_out=cmd.readline()
        cnum_cycles=cmd_out.split()[0]
        num_cycles=int(cnum_cycles) - 6

        filename=sldir+'/'+slaname

        header_attr,cols,data = _read_mesafile(filename,data_rows=num_cycles)

        self.cols        = cols
        self.header_attr = header_attr
        self.data        = data

    def get(self, str_name):
        """
        return a column of data with the name str_name.

        Parameters
        ----------
        str_name : string
            The name of the column as printed in history.data or
            star.log get the available columns from self.cols (where
            you replace self with the name of your instance

        """

        column_array = self.data[:,self.cols[str_name]-1].astype('float')
        return column_array

class star_log(history_data):
    """
    Class derived from history_data class (copy). Existing just (for
    compatibility reasons) for older mesa python scripts.

    """



# below are some utilities that the user typically never calls directly


def _read_mesafile(filename,data_rows=0,only='all'):
    """ private routine that is not directly called by the user"""
    f=open(filename,'r')
    vv=[]
    v=[]
    lines = []
    line  = ''
    for i in range(0,6):
        line = f.readline()
        lines.extend([line])

    hval  = lines[2].split()
    hlist = lines[1].split()
    header_attr = {}
#    for a,b in zip(hlist,hval):
#        header_attr[a] = float(b)
    for a, b in zip(hlist, hval):
        # Check if b contains only digits and at most one '.'
       if isinstance(b, str) and not b.replace('.', '', 1).isdigit():
           header_attr[a] = b  # Keep it as a string
       else:
           header_attr[a] = float(b)  # Convert to float


    if only is 'header_attr':
        return header_attr

    cols    = {}
    colnum  = lines[4].split()
    colname = lines[5].split()
    for a,b in zip(colname,colnum):
        cols[a] = int(b)

    data = []

    old_percent = 0
    for i in range(data_rows):
        # writing reading status
        percent = int(i*100/np.max([1, data_rows-1]))
        if percent >= old_percent + 5:
            sys.stdout.flush()
            sys.stdout.write("\r reading " + "...%d%%" % percent)
            old_percent = percent
        line = f.readline()
        v=line.split()
        try:
            vv=np.array(v,dtype='float64')
        except ValueError:
            for item in v:
                if item.__contains__('.') and not item.__contains__('E'):
                    v[v.index(item)]='0'
        data.append(vv)

    print(' \n')
    f.close()
    a=np.array(data)
    data = []
    return header_attr, cols, a


def _cleanstarlog(file_in):
    """
    cleaning history.data or star.log file, e.g. to take care of
    repetitive restarts.

    private, should not be called by user directly

    Parameters
    ----------
    file_in : string
        Typically the filename of the mesa output history.data or
        star.log file, creates a clean file called history.datasa or
        star.logsa.

    (thanks to Raphael for providing this tool)

    """

    file_out=file_in+'sa'
    f = open(file_in)
    lignes = f.readlines()
    f.close()

    nb    = np.array([],dtype=int)   # model number
    nb    = np.concatenate((nb    ,[  int(lignes[len(lignes)-1].split()[ 0])]))
    nbremove = np.array([],dtype=int)   # model number
    i=-1

    for i in np.arange(len(lignes)-1,0,-1):
        line = lignes[i-1]

        if i > 6 and line != "" :
            if int(line.split()[ 0])>=nb[-1]:
                nbremove = np.concatenate((nbremove,[i-1]))
            else:
                nb = np.concatenate((nb    ,[  int(line.split()[ 0])]))
    i=-1
    for j in nbremove:
        lignes.remove(lignes[j])

    fout = open(file_out,'w')
    for j in np.arange(len(lignes)):
        fout.write(lignes[j])
    fout.close()
