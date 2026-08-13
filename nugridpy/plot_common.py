#
# NuGridpy - Tools for accessing and visualising NuGrid data.
#
# Copyright 2007 - 2014 by the NuGrid Team.
# All rights reserved. See LICENSE.
#

"""
plot_common.py

Base plotting class shared by every NuGridpy data class (MESA and
nucleosynthesis-network alike): generic x-y plotting helpers, plus the
handful of methods (profile/density/abundance/Ye plots) that are
genuinely dual-concern and branch internally on whether they're
operating on a `mesa_profile` or an `se` instance.

Domain-specific plotting lives in `nugridpy.nu_plots` (nucleosynthesis
network output: PPN, mppnp/se) and `nugridpy.mesa_plots` (MESA stellar
evolution output), both of which subclass `PlotCommon`.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import zip
from builtins import str
from builtins import input
from builtins import range
from past.builtins import basestring
from builtins import object
from past.utils import old_div
from numpy import *
from math import *
import matplotlib.pylab as pyl
import matplotlib.pyplot as pl
#from matplotlib.mpl import colors,cm # depreciated in mpl ver 1.3
                                      # use line below instead
from matplotlib import colors,cm
import matplotlib
from matplotlib.patches import Rectangle, Arrow
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
from matplotlib.lines import Line2D
from matplotlib.ticker import *
from collections import OrderedDict
import numpy as np
import os
import os.path
import threading
import time
import sys

from . import astronomy as ast

def _padding_model_number(number, max_num):
    '''
    This method returns a zero-front padded string

    It makes out of str(45) -> '0045' if 999 < max_num < 10000. This is
    meant to work for reasonable integers (maybe less than 10^6).

    Parameters
    ----------
    number : integer
        number that the string should represent.
    max_num : integer
        max number of cycle list, implies how many 0s have be padded

    '''

    cnum = str(number)
    clen = len(cnum)

    cmax = int(log10(max_num)) + 1

    return (cmax - clen)*'0' + cnum


class PlotCommon(object):

    _classTest_data = {
        'ppm.yprofile': 'YProfile',
        'ascii_table.ascii_table': 'AsciiTable',
        'nugridse.se': 'se',
        'mesa.mesa_profile': 'mesa_profile',
        'mesa.star_log': 'mesa.star_log',
        'mesa.history_data': 'mesa.star_log',
        'ppn.xtime': 'xtime',
        'ppn.abu_vector': 'PPN',
        'starobs.plot': 'starobs',
        'grain.gdb': 'grain',
        }

    def _classTest(self):
        '''
        Determines what the type of class instance the subclass is, so
        we can dynamically determine the behaviour of methods.

        The data this method uses (_classTest_data) NEEDS to be
        modified if any names of files or classes are changed.

        TODO - The entire use of this class needs to be refactored to use
        derived classes instead.
        '''
        c = '.'.join(str(self.__class__)[:-2].rsplit('.', 2)[-2:])
        return self._classTest_data.get(c, '')

    def _which(self, program):
        '''
        Mimics which in the unix shell.

        '''
        def is_exe(fpath):
            return os.path.exists(fpath) and os.access(fpath, os.X_OK)

        fpath, fname = os.path.split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file

        return None

    def _logarithm(self, tmpX, tmpY, logX, logY, base):
        logXER=False
        logYER=False
        for i in range(len(tmpX)):
            if tmpX[i]<=0. and logX:
                #print 'We can not log a number less than or equal to zero'
                #print 'Attempting to remove incompatible values from X'
                logXER=True
            if tmpY[i]<=0. and logY:
                #print 'We can not log a number less than or equal to zero'
                #print 'Attempting to remove incompatible values from Y'
                logYER=True
        tmX=[]
        tmY=[]

        if logXER:
            for i in range(len(tmpX)):
                if tmpX[i]>0.:
                    tmX.append( tmpX[i])
                    tmY.append(tmpY[i])
            tmpX=tmX
            tmpY=tmY
        elif logYER:
            for i in range(len(tmpY)):
                if tmpY[i]>0.:
                    tmX.append( tmpX[i])
                    tmY.append(tmpY[i])
            tmpX=tmX
            tmpY=tmY

        tmX=tmpX
        tmY=tmpY

        if logX:
            tmX=tmpX
            try:
                for i in range(len(tmpX)):
                    tmX[i]=log(tmpX[i],base)
            except ValueError:
                #print 'We can not log a number less than or equal to zero'
                #print 'Attempting to remove incompatible values from X'
                logXER=True
        if logY:
            tmY=tmpY
            try:
                for i in range(len(tmpY)):
                    tmY[i]=log(tmpY[i],base)
            except ValueError:
                #print 'We can not log a number less than or equal to zero'
                #print 'Attempting to remove incompatible values from Y'
                logYER=True

        if logX:
            tmpX=tmX
        if logY:
            tmpY=tmY

        return tmpX,tmpY

    def _sparse(self, x, y, sparse):
        """
        Method that removes every non sparse th element.

        For example:
        if this argument was 5, This method would plot the 0th, 5th,
        10th ... elements.

        Parameters
        ----------
        x : list
            list of x values, of length j.
        y : list
            list of y values, of length j.
        sparse : integer
            Argument that skips every so many data points.

        """
        tmpX=[]
        tmpY=[]

        for i in range(len(x)):
            if sparse == 1:
                return x,y
            if (i%sparse)==0:
                tmpX.append(x[i])
                tmpY.append(y[i])
        return tmpX, tmpY

    def plotMulti(self, atrix, atriy, cyclist, title, path='/',
                  legend=None, labelx=None, labely=None, logx=False,
                  logy=False, base=10, sparse=1, pdf=False,
                  limits=None):
        '''
        Method for plotting multiple plots and saving it to multiple
        pngs or PDFs.

        Parameters
        ----------
        atrix : string
            The name of the attribute you want on the x axis.
        atriy : string
            The name of the attribute you want on the Y axis.
        cyclist : list
            List of cycles that you would like plotted.
        title : string
            The title of the graph and the name of the file.
        path : string, optional
            The file path. The default is '/'
        Legend : list or intager, optional
            A list of legends for each of your cycles, or one legend for
            all of the cycles. The default is None.
        labelx : string, optional
            The label on the X axis. The default is None.
        labely : string, optional
            The label on the Y axis. The default is None.
        logx : boolean, optional
            A boolean of whether the user wants the x axis
            logarithmically. The default is False.
        logy : boolean, optional
            A boolean of whether the user wants the Y axis
            logarithmically. The default is False.
        base : integer, optional
            The base of the logarithm. The default is 10.
        sparse : integer, optional
            Argument that skips every so many data points.  For example
            if this argument was 5, This method would plot the 0th,
            5th, 10th ... elements. The default is 1.
        pdf : boolean, optional
            A boolean of if the image should be saved to a pdf file.
            xMin, xMax, yMin, YMax:  plot coordinates.  The default is
            False.
        limits : list, optional
            The length four list of the x and y limits.  The order of
            the list is xmin, xmax, ymin, ymax. The default is None.

        '''
        if str(legend.__class__)!="<type 'list'>":# Determines the legend is a list
            legendList=False
        else:
            legendList=True

        if legendList and len(cyclist) !=len(legend): #if it is a list, make sure there is an entry for each cycle
            print('Please input a proper legend, with correct length, aborting plot')
            return None
        for i in range(len(cyclist)):
            if legendList:
                self.plot(atrix,atriy,cyclist[i],'ndump',legend[i],labelx,labely,base=base,sparse=sparse, \
                                  logx=logx,logy=logy,show=False,limits=limits)
            else:
                self.plot(atrix,atriy,cyclist[i],'ndump',legend,labelx,labely,base=base,sparse=sparse, \
                                  logx=logx,logy=logy,show=False,limits=limits)

            pl.title(title)
            if not pdf:
                currentDir = os.getcwd()
                os.chdir(path)
                pl.savefig(title+str(cyclist[i])+'.png', dpi=400)
                os.chdir(currentDir)
            else:
                currentDir = os.getcwd()
                os.chdir(path)
                pl.savefig(title+str(cyclist[i])+'.pdf', dpi=400)
                os.chdir(currentDir)
            pl.clf()
        return None

    def plot(self, atrix, atriy, fname=None, numtype='ndump',
             legend=None, labelx=None, labely=None, indexx=None,
             indexy=None, title=None, shape='.', logx=False,
             logy=False, path='/', base=10, sparse=1, show=True, pdf=False,limits=None,
             markevery=None, linewidth=1):
        """
        Simple function that plots atriy as a function of atrix

        This method will automatically find and plot the requested data.

        Parameters
        ----------
        atrix : string
            The name of the attribute you want on the x axis.
        atriy : string
            The name of the attribute you want on the Y axis.
        fname : optional
            Be the filename, Ndump or time, or cycle, If fname is a
            list, this method will then save a png for each cycle in the
            list.  Warning, this must be a list of cycles and not a
            list of filenames. The default is None.
        numtype : string, optional
            designates how this function acts and how it interprets
            fname. if numtype is 'file', this function will get the
            desird attribute from that file.  if numtype is 'NDump'
            function will look at the cycle with that nDump.  if numtype
            is 't' or 'time' function will find the _cycle with the
            closest time stamp. The default is 'ndump'.
        legend : list or intager, optional
            A list of legends for each of your cycles, or one legend for
            all of the cycles. The default is None.
        labelx : string, optional
            The label on the X axis. The default is None.
        labely : string, optional
            The label on the Y axis. The default is None.
        indexx : optional
            Depreciated: If the get method returns a list of lists,
            indexx would be the list at the index indexx in the list.
            The default is None.
        indexy : optional
            Depreciated: If the get method returns a list of lists,
            indexy would be the list at the index indexx in the list.
            The default is None.
        title : string, optional
            The Title of the Graph. The default is None.
        shape : string, optional
            What shape and colour the user would like their plot in.
            Please see
            http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.plot
            for all possible choices. The default is '.'.
        logx : boolean, optional
            A boolean of weather the user wants the x axi
            logarithmically. The default is False.
        logy : boolean, optional
            A boolean of weather the user wants the Y axis
            logarithmically. The default is False.
        path : string, optional
            Usef for PlotMulti, give the path where to save the Figures
        base : integer, optional
            The base of the logarithm. The Default is 10.
        sparse : integer, optional
            Argument that skips every so many data points. For example
            if this argument was 5, This method would plot the 0th, 5th,
            10th ... elements. The default is 1.
        show : boolean, optional
            A boolean of if the plot should be displayed useful with the
            multiPlot method. The default is True.
        pdf : boolean, optional
            PDF for PlotMulti? Default: False
        limits : list, optional
            The length four list of the x and y limits. The order of the
            list is xmin, xmax, ymin, ymax. The defautl is .
        markevery : integer or tupler, optional
            Set the markevery property to subsample the plot when
            using markers.  markevery can be None, very point will be
            plotted. It can be an integer N, Every N-th marker will be
            plotted starting with marker 0. It can be a tuple,
            markevery=(start, N) will start at point start and plot
            every N-th marker. The default is None.
        linewidth : integer, optional
            Set linewidth. The default is 1.

        Notes
        -----
        WARNING: Unstable if get returns a list with only one element (x=[0]).

        parameters: indexx and indexy have been deprecated.
        """
        t1=time.time()
        #Setting the axis labels

        if labelx== None :
            labelx=atrix
        if labely== None :
            labely=atriy

        if title!=None:
            title=title
        else:
            title=labely+' vs '+labelx

        if str(fname.__class__)=="<type 'list'>":
            self.plotMulti(atrix,atriy,fname,title,path,legend,labelx,labely,logx, logy, 10,1,pdf,limits)
            return
        tmpX=[]
        tmpY=[]
        singleX=False
        singleY=False
        #Getting data
        plotType=self._classTest()
        if plotType=='YProfile':
            if fname==None:
                fname=self.cycles[-1]

            listY=self.get(atriy,fname, numtype,resolution='a')
            listX=self.get(atrix,fname, numtype,resolution='a')
        elif plotType=='se':
            if fname==None:
                listY=self.get( atriy,sparse=sparse)
                listX=self.get(atrix,sparse=sparse)
            else:
                listY=self.get(fname, atriy,sparse=sparse)
                listX=self.get(fname, atrix,sparse=sparse)

            t2= time.time()
            print(t2 -t1)
        elif plotType=='PPN' :
            if fname==None and atrix not in self.cattrs and atriy not in self.cattrs:
                fname=len(self.files)-1
            if numtype=='ndump':
                numtype='cycNum'
            listY=self.get(atriy,fname,numtype)
            listX=self.get(atrix,fname,numtype)
        elif plotType=='xtime' or plotType=='mesa_profile' or plotType=='AsciiTable' or plotType=='mesa.star_log' or plotType=='starobs':
            listY=self.get(atriy)
            listX=self.get(atrix)
        else:
            listY=self.get(atriy)
            listX=self.get(atrix)
        tmpX=[]
        tmpY=[]
        if isinstance(listX[0], basestring) or isinstance(listY[0], basestring):
            for i in range(len(listX)):
                if '*****' == listX[i] or '*****' == listY[i]:
                    print('There seems to be a string of * in the lists')
                    print('Cutting out elements in both the lists that have an index equal to or greater than the index of the location of the string of *')
                    break
                tmpX.append(float(listX[i]))
                tmpY.append(float(listY[i]))

            listX=tmpX
            listY=tmpY




        #Determining if listX is a list or a list of lists
        try:
            j=listX[0][0]
        except:
            singleX = True

        if len(listX) == 1:  # If it is a list of lists with one element.
            tmpX=listX[0]
        elif singleX == True:# If it is a plain list of values.
            tmpX=listX
        elif indexx==None and len(listX)>1: # If it is a list of lists of values.
                                            # take the largest
            tmpX=listX[0]
            for i in range(len(listX)):
                if len(tmpX)<len(listX[i]):
                    tmpX=listX[i]
        elif indexx<len(listX): # If an index is specified, use that index
            tmpX=listX[indexx]
        else:
            print('Sorry that indexx does not exist, returning None')
            return None

        #Determining if listY is a list or a list of lists
        try:
            j=listY[0][0]
        except:
            singleY = True

        if len(listY) == 1: # If it is a list of lists with one element.
            #print 'hello'
            tmpY=listY[0]
        elif singleY == True: # If it is a plain list of values.
            #print 'world'
            tmpY=listY
        elif indexy==None and len(listY)>1:# If it is a list of lists of values.
                                            # take the largest
            #print 'fourth'
            tmpY=listY[0]
            for i in range(len(listY)):
                if len(tmpY)<len(listY[i]):
                    tmpY=listY[i]
        elif indexy<len(listY): # If an index is specified, use that index
            #print 'sixth'
            tmpY=listY[indexy]
        else:
            print('Sorry that indexy does not exist, returning None')
            return None
        '''
        elif indexy==None and len(listY)==1:
                #print 'fifth'
                tmpY=listY
        '''




        #Here, if we end up with different sized lists to plot, it
        #searches for a list that is of an equal length
        if len(tmpY)!=len(tmpX):
            found=False
            print("It seems like the lists are not of equal length")
            print("Now attempting to find a compatible list for ListX")
            for i in range(len(listY)):
                if not singleY and len(tmpX)==len(listY[i]):
                    tmpY=listY[i]
                    found=True

            if not found:
                print("Now attempting to find a compatible list for ListY")
                for i in range(len(listX)):

                    if not singleX and len(tmpY)==len(listX[i]):
                        tmpX=listX[i]
                        found=True

            if found:
                print("Suitable list found")
            else:

                print("There is no suitalble list, returning None")
                return None
        if len(tmpY)!=len(tmpX) and single == True:
            print('It seems that the selected lists are of different\nsize, now returning none')
            return None
        # Sparse stuff
        if plotType!='se':
            tmpX,tmpY=self._sparse(tmpX,tmpY, sparse)

        # Logarithm stuff
        if logy or logx:
            tmpX,tmpY=self._logarithm(tmpX,tmpY,logx,logy,base)

        # Here it ensures that if we are plotting ncycle no values of '*' will be plotted
        tmX=[]
        tmY=[]
        for i in range(len(tmpX)):
            tmX.append(str(tmpX[i]))
            tmY.append(str(tmpY[i]))

        tmpX=[]
        tmpY=[]
        for i in range(len(tmX)):
            if '*' in tmX[i] or '*' in tmY[i]:
                print('There seems to be a string of * in the lists')
                print('Cutting out elements in both the lists that have an index equal to or greater than the index of the location of the string of *')
                break
            tmpX.append(float(tmX[i]))
            tmpY.append(float(tmY[i]))
        listX=tmpX
        listY=tmpY

        #Setting the axis labels

        if logx:
            labelx='log '+labelx
        if logy:
            labely='log '+labely

        if legend!=None:
            legend=legend
        else:
            legend=labely+' vs '+labelx



        pl.plot(listX,listY,shape,label=legend,markevery=markevery,linewidth=linewidth)
        pl.legend()
        pl.title(title)
        pl.xlabel(labelx)
        pl.ylabel(labely)
        if show:
            pl.show()

        if limits != None and len(limits)==4:

            pl.xlim(limits[0],limits[1])
            pl.ylim(limits[2],limits[3])

    def _clear(self, title=True, xlabel=True, ylabel=True):
        '''
        Method for removing the title and/or xlabel and/or Ylabel.

        Parameters
        ----------
        Title : boolean, optional
            Boolean of if title will be cleared.  The default is True.
        xlabel : boolean, optional
            Boolean of if xlabel will be cleared.  The default is True.
        ylabel : boolean, optional
            Boolean of if ylabel will be cleared.  The default is True.

        '''
        if title:
            pyl.title('')
        if xlabel:
            pyl.xlabel('')
        if ylabel:
            pyl.ylabel('')

    # From mesa.py
    def _xlimrev(self):
        ''' reverse xrange'''
        xmax,xmin=pyl.xlim()
        pyl.xlim(xmin,xmax)

    def plot_prof_1(self, species, keystring, xlim1, xlim2, ylim1,
                    ylim2, symbol=None, show=False):
        '''
        Plot one species for cycle between xlim1 and xlim2 Only works
        with instances of se and mesa _profile.

        Parameters
        ----------
        species : list
            Which species to plot.
        keystring : string or integer
            Label that appears in the plot or in the case of se, a
            cycle.
        xlim1, xlim2 : integer or float
            Mass coordinate range.
        ylim1, ylim2 : integer or float
            Mass fraction coordinate range.
        symbol : string, optional
            Which symbol you want to use.  If None symbol is set to '-'.
            The default is None.
        show : boolean, optional
            Show the ploted graph.  The default is False.

        '''
        plotType=self._classTest()
        if plotType=='se':
            #tot_mass=self.se.get(keystring,'total_mass')
            tot_mass=self.se.get('mini')
            age=self.se.get(keystring,'age')
            mass=self.se.get(keystring,'mass')
            Xspecies=self.se.get(keystring,'iso_massf',species)

            mod=keystring
        elif plotType=='mesa_profile':
            tot_mass=self.header_attr['star_mass']
            age=self.header_attr['star_age']
            mass=self.get('mass')
            mod=self.header_attr['model_number']
            Xspecies=self.get(species)
        else:
            print('This method is not supported for '+str(self.__class__))
            return

        if symbol == None:
            symbol = '-'

        x,y=self._logarithm(Xspecies,mass,True,False,10)
        #print x
        pl.plot(y,x,symbol,label=str(species))
        pl.xlim(xlim1,xlim2)
        pl.ylim(ylim1,ylim2)
        pl.legend()

        pl.xlabel('$Mass$ $coordinate$', fontsize=20)
        pl.ylabel('$X_{i}$', fontsize=20)
        #pl.title('Mass='+str(tot_mass)+', Time='+str(age)+' years, cycle='+str(mod))
        pl.title('Mass='+str(tot_mass)+', cycle='+str(mod))
        if show:
            pl.show()

    def density_profile(self,ixaxis='mass',ifig=None,colour=None,label=None,fname=None):
        '''
        Plot density as a function of either mass coordiate or radius.

        Parameters
        ----------
        ixaxis : string
            'mass' or 'radius'
            The default value is 'mass'
        ifig : integer or string
            The figure label
            The default value is None
        colour : string
            What colour the line should be
            The default value is None
        label : string
            Label for the line
            The default value is None
        fname : integer
            What cycle to plot from (if SE output)
            The default value is None
        '''

        pT=self._classTest()

        # Class-specific things:
        if pT == 'mesa_profile':
            x = self.get(ixaxis)
            if ixaxis == 'radius':
                x = x*ast.rsun_cm
            y = self.get('logRho')
        elif pT == 'se':
            if fname is None:
                raise IOError("Please provide the cycle number fname")
            x = self.se.get(fname,ixaxis)
            y = np.log10(self.se.get(fname,'rho'))
        else:
            raise IOError("Sorry. the density_profile method is not available \
                          for this class")

        # Plot-specific things:
        if ixaxis == 'radius':
            x = np.log10(x)
            xlab='$\log_{10}(r\,/\,{\\rm cm})$'
        else:
            xlab='${\\rm Mass}\,/\,M_\odot$'

        if ifig is not None:
            pl.figure(ifig)
        if label is not None:
            if colour is not None:
                pl.plot(x,y,color=colour,label=label)
            else:
                pl.plot(x,y,label=label)
            pl.legend(loc='best').draw_frame(False)
        else:
            if colour is not None:
                pl.plot(x,y,color=colour)
            else:
                pl.plot(x,y)

        pl.xlabel(xlab)
        pl.ylabel('$\log_{10}(\\rho\,/\,{\\rm g\,cm}^{-3})$')

    def abu_profile(self,ixaxis='mass',isos=None,ifig=None,fname=None,logy=False,
                    colourblind=False):
        '''
        Plot common abundances as a function of either mass coordiate or radius.

        Parameters
        ----------
        ixaxis : string, optional
            'mass',  'logradius' or 'radius'
            The default value is 'mass'
        isos : list, optional
            list of isos to plot, i.e. ['h1','he4','c12'] for MESA or
            ['H-1','He-4','C-12'] for SE output. If None, the code decides
            itself what to plot.
            The default is None.
        ifig : integer or string, optional
            The figure label
            The default value is None
        fname : integer, optional
            What cycle to plot from (if SE output)
            The default value is None
        logy : boolean, optional
            Should y-axis be logarithmic?
            The default value is False
        colourblind : boolean, optional
            do you want to use the colourblind colour palette from the NuGrid
            nuutils module?
        '''

        pT=self._classTest()
        # Class-specific things:
        if pT == 'mesa_profile':
            x = self.get(ixaxis)
            if ixaxis == 'radius':
                x = x*ast.rsun_cm
            if isos is None:
                isos=['h1','he4','c12','c13','n14','o16','ne20','ne22','mg24','mg25',
                      'al26','si28','si30','s32','s34','cl35','ar36','ar38','cr52',
                      'cr56','fe56','ni56']
            risos=[i for i in isos if i in self.cols]
            abunds = [self.get(riso) for riso in risos]
            names=risos
        elif pT == 'se':
            if fname is None:
                raise IOError("Please provide the cycle number fname")
            x = self.se.get(fname,ixaxis)
            if isos is None:
                isos=['H-1','He-4','C-12','C-13','N-14','O-16','Ne-20','Ne-22','Mg-24','Mg-25',
                      'Sl-26','Si-28','Si-30','S-32','S-34','Cl-35','Ar-36','Ar-38','Cr-52',
                      'Cr-56','Fe-56','Ni-56']
            risos=[i for i in isos if i in self.se.isotopes]
            abunds = self.se.get(fname,'iso_massf',risos)
            names=risos
        else:
            raise IOError("Sorry. the density_profile method is not available \
                          for this class")

        # Plot-specific things:
        if ixaxis == 'logradius':
            x = np.log10(x)
            xlab='$\log_{10}(r\,/\,{\\rm cm})$'
        elif ixaxis == 'radius':
            x = old_div(x, 1.e8)
            xlab = 'r / Mm'
        else:
            xlab='${\\rm Mass}\,/\,M_\odot$'

        if ifig is not None:
            pl.figure(ifig)
        from . import utils as u
        cb = u.colourblind
        lscb = u.linestylecb # colourblind linestyle function
        for i in range(len(risos)):
            if logy:
                y = np.log10(abunds if len(risos) < 2 else abunds[i])
            else:
                y = abunds if len(risos) < 2 else abunds[i]
            if colourblind:
                pl.plot(x,y,ls=lscb(i)[0],marker=lscb(i)[1],
                        color=lscb(i)[2],markevery=u.linestyle(i)[1]*20,
                        label=names[i],mec='None')
            else:
                pl.plot(x,y,u.linestyle(i)[0],markevery=u.linestyle(i)[1]*20,
                        label=names[i],mec='None')

        pl.legend(loc='best').draw_frame(False)
        pl.xlabel(xlab)
        pl.ylabel('$\log(X)$')

    def ye_profile(self,ixaxis='mass',ifig=None,colour=None,label=None,fname=None):
        '''
            Plot electron fraction Y_e as a function of either mass coordiate or radius.

            Parameters
            ----------
            ixaxis : string
            'mass' or 'radius'
            The default value is 'mass'
            ifig : integer or string
            The figure label
            The default value is None
            colour : string
            What colour the line should be
            The default value is None
            label : string
            Label for the line
            The default value is None
            fname : integer
            What cycle to plot from (if SE output)
            The default value is None
            '''

        pT=self._classTest()

        # Class-specific things:
        if pT == 'mesa_profile':
            x = self.get(ixaxis)
            if ixaxis == 'radius':
                x = x*ast.rsun_cm
            y = self.get('ye')
        elif pT == 'se':
            if fname is None:
                raise IOError("Please provide the cycle number fname")
            raise IOError("Sorry, Ye profile is not yet available for\
                          the SE data class")
        #            x = self.se.get(fname,ixaxis)
        #            y = None # not implemented yet...
        else:
            raise IOError("Sorry. the density_profile method is not available \
                          for this class")

        # Plot-specific things:
        if ixaxis == 'radius':
            x = np.log10(x)
            xlab='$\log_{10}(r\,/\,{\\rm cm})$'
        else:
            xlab='${\\rm Mass}\,/\,M_\odot$'

        if ifig is not None:
            pl.figure(ifig)
        if label is not None:
            if colour is not None:
                pl.plot(x,y,color=colour,label=label)
            else:
                pl.plot(x,y,label=label)
            pl.legend(loc='best').draw_frame(False)
        else:
            if colour is not None:
                pl.plot(x,y,color=colour)
            else:
                pl.plot(x,y)

        pl.xlabel(xlab)
        pl.ylabel('$Y_{\\rm e}$')


    # From mesa.star_log

