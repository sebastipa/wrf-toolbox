import re
import os
import shutil
import math
import pyproj
import cartopy.crs as crs
import cartopy.feature as cfeature
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# wrfpre.py   : a Python module to create WRF case directories, set up namelist files and plot WPS domains
# classes     : wrfcase, wps
# author      : Sebastiano Stipa
# date        : 27-11-2024

# ===============================================================================================
# Class wrfcase definition  : creates a WRF case directory structure and sets up namelist files
# case_init                 : creates the case directory structure and for an em_real simulation 
#                             and copies suitable WRF executables and files
# create_namelist_wps       : copies namelist.wps from current folder into the case folder and 
#                             overwirtes start and end dates 
# create_namelist_wrf       : copies namelist.input from current folder into the case folder and 
#                             overwirtes start and end dates 
# set_entry                 : sets an entry in a namelist file
# remove_entry              : removes an entry from a namelist file
# create_turbines_wrf       : creates a file with the turbine locations and adds the entry to
#                             the namelist.input file
# wps_setup                 : links forcing data and sets up variable table, preparing the case 
#                             for geogrid, ungrib and metgrid
# wrf_setup                 : links WPS files to WRF directory, preparing the case for real.exe
#                             and wrf.exe
# ===============================================================================================

class wrfcase:
    
    project       = 'project_name'                                            # project of which the case is part of 
    creation_date = 'case_creation_date'                                      # date of case creation
    name          = 'case_full_name'                                          # full name of the case
    author        = 'author_name'                                             # author of the case
    forcing_type  = 'forcinhg_data_type'                                      # type of forcing data used in the case (eg. 'era5' or 'gfs')
    case_dir      = 'case_directory'                                          # name of the case directory
    case_path     = '/absolute_path_to_case_directory/'                       # path to the case directory (excluding case directory)
    case_root     = '/absolute_path_to_case_directory/case_dir/'              # path to the case directory (including case directory)

    data_dir      = '/absolute_path_to_case_directory/case_dir/forcing_type/' # path to the forcing data directory
    wps_dir       = '/absolute_path_to_case_directory/case_dir/WPS/'          # path to the WPS directory
    wrf_dir       = '/absolute_path_to_case_directory/case_dir/WRF/'          # path to the WRF directory
    
    def __init__(self, 
                 project, 
                 creation_date, 
                 name, 
                 author, 
                 forcing_type,
                 case_dir, 
                 case_path):
        
        self.project      = project
        self.creation_date= creation_date
        self.name         = name
        self.author       = author
        self.forcing_type = forcing_type
        self.case_dir     = self.adjust_filedirname(case_dir)
        self.case_path    = self.adjust_paths(case_path)
        self.case_root    = self.adjust_paths(self.case_path + self.case_dir)
        self.data_dir     = self.adjust_paths(self.case_root + self.forcing_type.upper())
        self.wps_dir      = self.adjust_paths(self.case_root + "WPS")
        self.wrf_dir      = self.adjust_paths(self.case_root + "WRF")

        # check that datatype is a string 
        if(not isinstance(forcing_type, str)):
            raise Exception("forcing_type must be a string: eg. 'era5' or 'gfs'\n")

        return
    
    # add starting and trailing '/' if not present
    def adjust_paths(self, path):

        if(path[0] != '/'):
            path = '/' + path
        
        if(path[-1] != '/'):
            path = path + '/'
        
        return path
    
    # remove starting and trailing '/' from filename if present
    def adjust_filedirname(self, filename):

        if(filename[0] == '/'):
            filename = filename[1:]

        if(filename[-1] == '/'):
            filename = filename[:-1]

        return filename
    
    # creates the case directory structure and copies WRF executables and files
    def case_init(self, wrf_install_dir, wps_install_dir):
        
        wrf_install_dir = self.adjust_paths(wrf_install_dir)
        wps_install_dir = self.adjust_paths(wps_install_dir)	

        # create case directory
        os.makedirs(self.case_path + self.case_dir, exist_ok=True)

        # create subdiurectories for each phase of the WRF workflow 
        os.makedirs(self.case_root + self.forcing_type.upper(), exist_ok=True)
        os.makedirs(self.case_root + "WPS", exist_ok=True)
        os.makedirs(self.case_root + "WRF", exist_ok=True)

        # setup WRF directory
        shutil.copy(wrf_install_dir + 'test/em_real/wrf.exe', self.case_root + "WRF/wrf.exe")
        shutil.copy(wrf_install_dir + 'test/em_real/real.exe', self.case_root + "WRF/real.exe")
        shutil.copy(wrf_install_dir + 'test/em_real/CAMtr_volume_mixing_ratio', self.case_root + "WRF/CAMtr_volume_mixing_ratio")
        shutil.copy(wrf_install_dir + 'test/em_real/GENPARM.TBL', self.case_root + "WRF/GENPARM.TBL")
        shutil.copy(wrf_install_dir + 'test/em_real/LANDUSE.TBL', self.case_root + "WRF/LANDUSE.TBL")
        shutil.copy(wrf_install_dir + 'test/em_real/SOILPARM.TBL', self.case_root + "WRF/SOILPARM.TBL")
        shutil.copy(wrf_install_dir + 'test/em_real/VEGPARM.TBL', self.case_root + "WRF/VEGPARM.TBL")
        shutil.copy(wrf_install_dir + 'test/em_real/RRTMG_LW_DATA', self.case_root + "WRF/RRTMG_LW_DATA")
        shutil.copy(wrf_install_dir + 'test/em_real/RRTMG_SW_DATA', self.case_root + "WRF/RRTMG_SW_DATA")
        shutil.copy(wrf_install_dir + 'test/em_real/ozone.formatted', self.case_root + "WRF/ozone.formatted")
        shutil.copy(wrf_install_dir + 'test/em_real/ozone_lat.formatted', self.case_root + "WRF/ozone_lat.formatted")
        shutil.copy(wrf_install_dir + 'test/em_real/ozone_plev.formatted', self.case_root + "WRF/ozone_plev.formatted")

    	# create results and postprocessing directories
        os.makedirs(self.case_root + "WRF/results", exist_ok=True)
        os.makedirs(self.case_root + "WRF/postprocessing", exist_ok=True)

        # setup WPS directory
        os.makedirs(self.case_root + "WPS/metgrid", exist_ok=True)
        os.makedirs(self.case_root + "WPS/geogrid", exist_ok=True)
        os.makedirs(self.case_root + "WPS/ungrib", exist_ok=True)
        shutil.copy(wps_install_dir + 'metgrid.exe', self.case_root + "WPS/metgrid.exe")
        shutil.copy(wps_install_dir + 'ungrib.exe', self.case_root + "WPS/ungrib.exe")
        shutil.copy(wps_install_dir + 'geogrid.exe', self.case_root + "WPS/geogrid.exe")
        shutil.copy(wps_install_dir + 'link_grib.csh', self.case_root + "WPS/link_grib.csh")
        shutil.copy(wps_install_dir + 'metgrid/METGRID.TBL', self.case_root + "WPS/metgrid/METGRID.TBL")
        shutil.copy(wps_install_dir + 'geogrid/GEOGRID.TBL', self.case_root + "WPS/geogrid/GEOGRID.TBL")
        shutil.copytree(wps_install_dir + 'ungrib/Variable_Tables', self.case_root + "WPS/ungrib/Variable_Tables",  dirs_exist_ok=True)
         
        return
    
    # copies namelist.wps from current directory to case directory overwriting start and end dates
    def create_namelist_wps(self, 
                            start_year, start_month, start_day, start_hour,
                            end_year,   end_month,   end_day,   end_hour):
        
        shutil.copy('namelist.wps', self.case_root + "WPS/namelist.wps")

        # overwrite start and end dates
        self.set_entry('namelist.wps', 'share', 'start_date',
                       str(start_year) + '-' + 
                       str(start_month).zfill(2) + '-' + 
                       str(start_day).zfill(2) + '_' + 
                       str(start_hour).zfill(2) + ':00:00')
        
        self.set_entry('namelist.wps', 'share', 'end_date',
                          str(end_year) + '-' + 
                          str(end_month).zfill(2) + '-' + 
                          str(end_day).zfill(2) + '_' + 
                          str(end_hour).zfill(2) + ':00:00')
        return
    
    # copies namelist.input from current directory to case directory overwriting start and end dates
    def create_namelist_wrf(self,
                            start_years, start_months, start_days, start_hours,
                            end_years,   end_months,   end_days,   end_hours):
        
        shutil.copy('namelist.input', self.case_root + "WRF/namelist.input")

        # create a string to be added as an entry 
        start_years_str  = ', '.join(map(str, start_years)) + ','
        start_months_str = ', '.join(map(str, start_months)) + ','   
        start_days_str   = ', '.join(map(str, start_days)) + ','       
        start_hours_str  = ', '.join(map(str, start_hours)) + ','     
        end_years_str    = ', '.join(map(str, end_years)) + ','     
        end_months_str   = ', '.join(map(str, end_months)) + ','           
        end_days_str     = ', '.join(map(str, end_days)) + ','       
        end_hours_str    = ', '.join(map(str, end_hours)) + ',' 

        # overwrite start and end dates for each domain
        self.set_entry('namelist.input', 'time_control', 'start_year', start_years_str)
        self.set_entry('namelist.input', 'time_control', 'start_month',start_months_str)
        self.set_entry('namelist.input', 'time_control', 'start_day',  start_days_str)
        self.set_entry('namelist.input', 'time_control', 'start_hour', start_hours_str)
        self.set_entry('namelist.input', 'time_control', 'end_year',   end_years_str)
        self.set_entry('namelist.input', 'time_control', 'end_month',  end_months_str)
        self.set_entry('namelist.input', 'time_control', 'end_day',    end_days_str)
        self.set_entry('namelist.input', 'time_control', 'end_hour',   end_hours_str)

        # remove run time entries 
        self.remove_entry('namelist.input', 'domains', 'run_days')
        self.remove_entry('namelist.input', 'domains', 'run_hours')
        self.remove_entry('namelist.input', 'domains', 'run_minutes')
        self.remove_entry('namelist.input', 'domains', 'run_seconds')

        return

    # removes an entry from a namelist file
    def remove_entry(self, file, section, keyword):

        def starts_with_keyword(line, keyword):
            pattern = r'^\s*' + re.escape(keyword.strip()) + r'\b'
            return re.match(pattern, line) is not None 
            
        if file == 'namelist.input':
            file_path = self.case_root + "/WRF/namelist.input"
        elif file == 'namelist.wps':
            file_path = self.case_root + "/WPS/namelist.wps"
        else:
            raise Exception("file must be 'namelist.input' or 'namelist.wps'\n")

        with open(file_path, 'r') as file:
            lines = file.readlines()

        has_section = False

        with open(file_path, 'w') as file:
            
            for line in lines:

                # check if we are in the section
                if starts_with_keyword(line, '&' + section):
                    has_section = True
                
                # check if section ended
                if has_section and starts_with_keyword(line, '/'):
                    has_section = False
                
                # if we are in the section, remove the entry
                if has_section and starts_with_keyword(line, keyword):
                    continue
                else:
                    file.write(line)    
            
        return

    # Sets an entry in a namelist file. If the entry is not found, it is added to the section. 
    # If the section is not found, it is created and the entry is added.
    def set_entry(self, file, section, keyword, entry):

        def starts_with_keyword(line, keyword):
            pattern = r'^\s*' + re.escape(keyword.strip()) + r'\b'
            return re.match(pattern, line) is not None

        formatted_entry = ' ' + (keyword + ' ').ljust(36) + '= ' + entry

        if file == 'namelist.input':
            file_path = self.case_root + "WRF/" + file
        elif file == 'namelist.wps':
            file_path = self.case_root + "WPS/" + file  
        else:
            raise Exception("file must be 'namelist.input' or 'namelist.wps'\n")

        with open(file_path, 'r') as file:
            lines = file.readlines()

        has_section = False
        entry_added = False

        with open(file_path, 'w') as file:
            
            for line in lines:

                # check if we are in the section
                if starts_with_keyword(line, '&' + section):
                    has_section = True
                
                # check if section ended
                if has_section and starts_with_keyword(line, '/'):
                    
                    # no entry found in section: add it
                    if(not entry_added):
                        file.write(formatted_entry + '\n')
                        entry_added = True
                    
                    # close section
                    file.write(line)
                    has_section = False
                    
                    # avoid writing closing line twice
                    continue
                
                # if we are in the section, write the entry
                if has_section and starts_with_keyword(line, keyword):
                    file.write(formatted_entry + '\n')
                    entry_added = True
                else:
                    file.write(line)    
            
            # if there was no section, create it and write the entry 
            if(not has_section and not entry_added):
                file.write('&' + section + '\n')
                file.write(formatted_entry + '\n')
                file.write('/\n')

        return
    
    def create_turbines_wrf(self, path_to_turbines):
        return
    
    # links datasets to WPS folder and sets up suitable variable table
    def wps_setup(self, var_table='Vtable.ERA-interim.pl', link_grib='link_grib.csh'):
        
        # copy the variable table to the WPS folder
        shutil.copy(self.wps_dir + 'ungrib/Variable_Tables/' + var_table, self.wps_dir + 'Vtable')	

        # save current path 
        original_directory = os.getcwd()

        # link the forcing data to the WPS folder
        os.chdir
        os.chdir(self.wps_dir)
        os.system(f'./{link_grib} {self.data_dir}')
        os.chdir(original_directory)

        return
    
    # links WPS files to WRF directory
    def wrf_setup(self):
            
        # save current path 
        original_directory = os.getcwd()
            
        os.chdir(self.wrf_dir)
        os.system(f'ln -sf {self.wps_dir}met_em.* .')
        os.system(f'ln -sf {self.wps_dir}geo_em.* .')
        os.chdir(original_directory)
        
        return

# ===============================================================================================
# Class wps definition: reads namelist.wps and provides methods to get setup info and plot domains
# get_bounds                : returns the bounds of a domain in projected coordinates as (x_min, x_max, y_min, y_max)
# get_proj                  : returns the projection of the domain as a cartopy.crs object
# add_domain_rectangle      : adds a rectangle to a plot representing a domain 
# plot_domains              : plots the domains of the WPS setup
# ===============================================================================================

class wps:

    filename          = "namelist.wps"

    ndomains          =   1
    parent_id         =   1
    parent_grid_ratio =   1
    i_parent_start    =   1
    j_parent_start    =   1
    e_we              =  91
    e_sn              =  100
    geog_data_res     = 'default'
    dx                = 27000
    dy                = 27000
    map_proj          = 'mercator'
    ref_lat           =  28.00
    ref_lon           = -75.00
    truelat1          =  30.0
    truelat2          =  60.0
    stand_lon         = -75.0

    def __init__(self, verbose=False):
        
        self.ndomains = self.__get_int_keyword("max_dom")[0]
        if verbose: print("ndomains          =", self.ndomains)

        self.parent_id = self.__get_int_keyword("parent_id")
        if verbose: print("parent_id         =", self.parent_id)

        self.parent_grid_ratio = self.__get_int_keyword("parent_grid_ratio")
        if verbose: print("parent_grid_ratio =", self.parent_grid_ratio)

        self.i_parent_start = self.__get_int_keyword("i_parent_start")
        if verbose: print("i_parent_start    =", self.i_parent_start)

        self.j_parent_start = self.__get_int_keyword("j_parent_start")
        if verbose: print("j_parent_start    =", self.j_parent_start)

        self.e_we = self.__get_int_keyword("e_we")
        if verbose: print("e_we              =", self.e_we)

        self.e_sn = self.__get_int_keyword("e_sn")
        if verbose: print("e_sn              =", self.e_sn)

        self.geog_data_res = self.__get_str_keyword("geog_data_res")
        if verbose: print("geog_data_res     =", self.geog_data_res)

        self.dx = self.__get_flt_keyword("dx")
        if verbose: print("dx                =", self.dx)

        self.dy = self.__get_flt_keyword("dy")
        if verbose: print("dy                =", self.dy)

        self.map_proj = self.__get_str_keyword("map_proj")
        if verbose: print("map_proj          =", self.map_proj)

        self.ref_lat = self.__get_flt_keyword("ref_lat")
        if verbose: print("ref_lat           =", self.ref_lat)

        self.ref_lon = self.__get_flt_keyword("ref_lon")
        if verbose: print("ref_lon           =", self.ref_lon)

        self.truelat1 = self.__get_flt_keyword("truelat1")
        if verbose: print("truelat1          =", self.truelat1)

        self.truelat2 = self.__get_flt_keyword("truelat2")
        if verbose: print("truelat2          =", self.truelat2)

        self.stand_lon = self.__get_flt_keyword("stand_lon")
        if verbose: print("stand_lon         =", self.stand_lon)

    # Private members ---------------------------------------------------------

    def __get_int_keyword(self, keyword):

        with open(self.filename, 'r') as file:
            for line in file:
                match = re.search(keyword, line)
                if match:
                    string = re.sub(keyword,"", line)
                    return([int(s) for s in re.findall(r'\b\d+\b', string)])
            
            raise Exception("did not find " + keyword + " keyword\n")
        
    def __get_flt_keyword(self, keyword):

        with open(self.filename, 'r') as file:
            for line in file:
                match = re.search(keyword, line)
                if match:
                    string = re.sub(keyword,"", line)
                    return([float(s) for s in re.findall(r"[-+]?(?:\d*\.*\d+)", string)])
            
            raise Exception("did not find " + keyword + " keyword\n")
        
    def __get_str_keyword(self, keyword):

        with open(self.filename, 'r') as file:
            for line in file:
                match = re.search(keyword, line)
                if match:
                    string = re.sub(keyword,"", line)
                    return(re.findall(r'\w+(?:-\w+)*', string))
            
            raise Exception("did not find " + keyword + " keyword\n")

    # Public members ----------------------------------------------------------

    def get_proj(self):

        match self.map_proj[0]:
            case "lambert":
                return(crs.LambertConformal(central_longitude=self.stand_lon[0],standard_parallels=(self.truelat1[0], self.truelat2[0])))                       
            case "mercator":
                return(crs.Mercator(central_longitude=self.ref_lon[0], latitude_true_scale=self.truelat1[0]))
            case "polar":
                return(crs.Stereographic(central_longitude=self.stand_lon[0], central_latitude=self.ref_lat[0], true_scale_latitude=self.truelat1[0]))
            case "lat-lon":
                return(crs.PlateCarree(central_longitude=self.ref_lon[0]))
            case _:
                raise Exception("Unknown map projection\n")
            
    def get_bounds(self, domain_id=1):
    
        if self.map_proj[0]=="lat-lon":
            meters2lat = 1.0 / 111320.0
            meters2lon = 1.0 / (111320.0 * math.cos(math.radians(self.ref_lat[0])))
        else:
            meters2lat = 1.0
            meters2lon = 1.0

        transformer_to_proj = pyproj.Transformer.from_crs("EPSG:4326", self.get_proj(), always_xy=True)
        x_center, y_center = transformer_to_proj.transform(self.ref_lon[0], self.ref_lat[0])

        # domain 1 (outer domain)
        x_min = x_center - (self.dx[0]*(self.e_we[0])/2.0) * meters2lon
        x_max = x_center + (self.dx[0]*(self.e_we[0])/2.0) * meters2lon
        y_min = y_center - (self.dy[0]*(self.e_sn[0])/2.0) * meters2lat
        y_max = y_center + (self.dy[0]*(self.e_sn[0])/2.0) * meters2lat

        # find the parent domain of the requested domain by iterating through the parent_id list.
        # While the parent id is different from one, we proceed iteratively by finding the parent domain of 
        # the current domain until that is equal to 1 and create an array of domain ids from the inner to the outer. 
        # Then work up that list from the outer domain until we have the bounds of the requested inner domain.

        if domain_id > self.ndomains:
            raise Exception("Requested domain ID is out of bounds\n")

        if domain_id > 1:

            # list of consecutive parents to the requested domain  
            nesting_list = [domain_id]

            child = domain_id
            while True:
                parent = self.parent_id[child-1]
                nesting_list.append(parent)
                if parent == 1:
                    break
                child = parent
                    
            nesting_list.reverse()

            dx = self.dx[0]
            dy = self.dy[0]

            # loop through the nesting list to find the bounds of the requested domain
            for d in range(1, len(nesting_list)):

                child  = nesting_list[d] - 1
                parent = self.parent_id[nesting_list[d]-1] - 1

                # child minimum bounds: use dx of parent from the parent minimum bounds
                x_min = x_min + dx*self.i_parent_start[child] * meters2lon
                y_min = y_min + dy*self.j_parent_start[child] * meters2lat

                # compute dx and dy of child from the parent dx and dy
                dx = dx / self.parent_grid_ratio[child]
                dy = dy / self.parent_grid_ratio[child]

                # child maximum bounds: use dx of child from the child minimum bounds
                x_max = x_min + (self.e_we[child]*dx) * meters2lon
                y_max = y_min + (self.e_sn[child]*dy) * meters2lat

        return(x_min, x_max, y_min, y_max)
    
    def add_domain_rectangle(self, ax, domain_id=1, color='black'):
            
            # get projection 
            proj = self.get_proj()

            # get domain bounds
            x_min, x_max, y_min, y_max = self.get_bounds(domain_id)
    
            # Create a rectangle in the Mercator projection's coordinates
            rectangle = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=2, edgecolor=color, facecolor='none', transform=proj)
            
            ax.add_patch(rectangle)

            ax.text(x_min + (x_max - x_min) / 5, y_max+5e4, "D " + str(domain_id), color=color,
                    horizontalalignment='center', verticalalignment='center', transform=proj, fontsize=12)
            
    def plot_domains(self):

        # Create the plot with Mercator projection
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': self.get_proj()})

        # Add domains a rectangles to the plot
        for i in range(1, self.ndomains+1):
            self.add_domain_rectangle(ax, i, color='black')

        # Set the extent of the plot close to the rectangle borders
        x_min, x_max, y_min, y_max = self.get_bounds(1)
        buffer_x = (x_max - x_min)*0.1
        buffer_y = (y_max - y_min)*0.1
        ax.set_xlim(left=x_min-buffer_x, right=x_max+buffer_x)
        ax.set_ylim(top=y_max+buffer_y, bottom=y_min-buffer_y)

        # Display gridlines
        ax.gridlines(draw_labels=True)

        # Add features 
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')  # Land
        ax.add_feature(cfeature.OCEAN, edgecolor='black', facecolor='lightblue')  # Ocean
        ax.add_feature(cfeature.BORDERS, edgecolor='black')  # Country borders
        ax.set_title("Domain setup using " + self.map_proj[0] + " projection")

        # Show the plot
        plt.show()

    
    