import metview as mv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import pandas as pd
import seaborn as sns
from   datetime import datetime, timedelta
from   scipy.optimize import curve_fit

# gribviz.py  : a Python module to help in WRF simulation preprocessing by extracting statistics 
#               and visualizing data from grib files
# classes     : gribdata
# author      : Sebastiano Stipa
# date        : 27-11-2024

class gribdata:

    """
    A class to represent and manipulate GRIB data for WRF simulation preprocessing.

    Attributes
    ----------
    day_range : list of int
        The range of days to process.
    month_range : list of int
        The range of months to process.
    year_range : list of int
        The range of years to process.
    prefix : str
        The prefix for the filenames.
    extension : str
        The extension for the filenames.
    path : str
        The path to the directory containing the GRIB files.
    """

    # dummy initialization
    start_day     = 1
    end_day       = 1
    start_month   = 1
    end_month     = 1
    start_year    = 2022
    end_year      = 2022
    path          = "/storage/wrf/nobackup/munters/ERA5"
    prefix        = "levels"
    extension     = "grib"
    
    # constants 
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monts_in_year = 12 

    # class constructor
    def __init__(self, day_range =[1, 1], 
                 month_range     =[1, 1], 
                 year_range      =[2022, 2022],
                 prefix          ="levels", 
                 extension       ="grib", 
                 path            ="/storage/wrf/nobackup/munters/ERA5/"
                 ):
        
        """
        Constructs all the necessary attributes for the gribdata object.

        Parameters
        ----------
        day_range : list of int, optional
            The range of days to process (default is [1, 1]).
        month_range : list of int, optional
            The range of months to process (default is [1, 1]).
        year_range : list of int, optional
            The range of years to process (default is [2022, 2022]).
        prefix : str, optional
            The prefix for the filenames (default is "levels"). :noindex:
        extension : str, optional
            The extension for the filenames (default is "grib"). :noindex:
        path : str, optional
            The path to the directory containing the GRIB files (default is "/storage/wrf/nobackup/munters/ERA5/"). :noindex:
        """

        # check that starting date exists 
        try:
            datetime(year_range[0], month_range[0], day_range[0])
            start_exists = True
        except ValueError:
            start_exists = False
        try:
            datetime(year_range[1], month_range[1], day_range[1])
            end_exists = True
        except ValueError:
            end_exists = False

        if not start_exists:
            raise ValueError("Invalid start date")
        if not end_exists:  
            raise ValueError("Invalid end date")

        if datetime(year_range[0], month_range[0], day_range[0]) > datetime(year_range[1], month_range[1], day_range[1]):
            raise ValueError("Start date is after end date")
        
        # assign values to class variables
        self.start_day   = day_range[0]
        self.end_day     = day_range[1]
        self.start_month = month_range[0]
        self.end_month   = month_range[1]
        self.start_year  = year_range[0]
        self.end_year    = year_range[1]
        self.path        = path
        self.prefix      = prefix
        self.extension   = extension	
        return

    # return an array of filenames ordered by date 
    def get_filenames(self):

        """
        Get the list of filenames ordered by date.

        Returns
        -------
        list of str
            The list of filenames ordered by date.
        """

        filenames = []
        start    = datetime(self.start_year, self.start_month, self.start_day)
        end      = datetime(self.end_year, self.end_month, self.end_day)

        while start <= end:
            year  = start.year
            month = start.month
            day   = start.day
            start += timedelta(days=1)
            filenames.append(self.path + "/" + str(year) + "/" + self.prefix + "_" + str(year) + "_" + str(month).zfill(2) + "_" + str(day).zfill(2) + ".grib")
               
        return filenames  
    
    def copy_files_to_dir(self, dest_dir):

        """
        Copy selected files into the destination folder (when creating WRF case).

        Parameters
        ----------
        dest_dir : str
            The destination directory where the files will be copied.
        """

        filenames = self.get_filenames()
        for filename in filenames:
            print(f"Copying {filename} --> {dest_dir}")
            os.system(f"cp {filename} {dest_dir}")
        return
    
    def link_files_to_dir(self, dest_dir):

        """
        Link selected files into the destination folder (when creating WRF case).

        Parameters
        ----------
        dest_dir : str
            The destination directory where the files will be linked.
        """
        filenames = self.get_filenames()
        for filename in filenames:
            print(f"Linking {filename} --> {dest_dir}")
            os.system(f"ln -s {filename} {dest_dir}")
        return
    
    # inspect the data in the grib files
    def inspect(self):

        """
        Inspect the current dataset in the GRIB files and print information.
        """

        filenames = self.get_filenames()
        fs        = mv.read(filenames[0])
        df        = fs.ls(filter={"step":0, "time":0}, extra_keys=['name', 'units'])
        print(df.to_markdown())
        return

    # compute min max lat and lon 
    def get_min_max_lat_lon(self, verbose=False):

        """
        Compute the minimum and maximum latitude and longitude of the current dataset.

        Returns
        -------
        tuple
            The minimum and maximum latitude and longitude as (min_lat, max_lat, min_lon, max_lon).
        """

        filenames = self.get_filenames()
        g         = mv.read(filenames[0])
        
        if(self.prefix == "levels"):
            f     = mv.read(data=g, param = 'r', levelist=500)
        elif(self.prefix == "single"):
            f     = mv.read(data=g, param = 'sp')
        else:
            raise ValueError("Invalid prefix")
            
        lats = mv.latitudes(f)
        lons = mv.longitudes(f)
        
        max_lat = max(lats[0,:])
        min_lat = min(lats[0,:])
        max_lon = max(lons[0,:])
        min_lon = min(lons[0,:])

        if verbose:
            print(f"min_lat: {min_lat}, max_lat: {max_lat}, min_lon: {min_lon}, max_lon: {max_lon}")
         
        return [min_lat, max_lat, min_lon, max_lon]  

    # get the elevation at a given lat, lon (it doesn't work very well)
    def get_elevation(self, lat=0, lon=0):

        """
        Get the elevation at a given latitude and longitude (needs more testing).

        Parameters
        ----------
        lat : float
            The latitude of the location.
        lon : float
            The longitude of the location.

        Returns
        -------
        float
            The elevation at the given latitude and longitude.
        """
        
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = requests.get(url)

        try:
            if response.status_code == 200:
                results = response.json().get('results')
                if results:
                    elevation = results[0].get('elevation')
                    return elevation
                else:
                    raise ValueError("No elevation data found for the given location.")
            else:
                raise ConnectionError(f"Failed to connect to the API. Status code: {response.status_code}")

        except Exception as err:
            print(err)
        
    
    # extract the time series of a wind speed at a given pressure level
    def windspeed_extract_timeseries_lvl(self, level="500"):

        """
        Extract the time series of wind speed at a given pressure level.

        Parameters
        ----------
        level : str
            The pressure level to extract.

        Returns
        -------
        list of tuple
            The time series and wind speed data.
        """

        if(self.prefix == "levels"):
            filenames = self.get_filenames()
            data = []
            for filename in filenames:
                if mv.exist(filename):
                    g    = mv.read(filename)
                    u    = mv.read(data=g, param="u", levelist=level)
                    v    = mv.read(data=g, param="v", levelist=level)
                    ws   = mv.sqrt(u*u + v*v)
                    ws   = mv.grib_set_long(ws, ['paramId', 10])
                else:
                    raise ValueError("Cannot open file " + filename) 

                times = mv.valid_date(ws)
                
                for tv in zip(times, ws):
                    data.append(tv)
            
            return data
        else:
            raise ValueError("windspeed_extract_timeseries_lvl only supports prefix=levels")
        
    # extract the time series of wind speed at 10m
    def windspeed_extract_timeseries_10m(self):

        """
        Extract the time series of wind speed at 10m.

        Returns
        -------
        list of tuple
            The time series and wind speed data.
        """

        if(self.prefix == "single"):
            filenames = self.get_filenames()
            data = []
            for filename in filenames:
                if mv.exist(filename):
                    g    = mv.read(filename)
                    u    = mv.read(data=g, param="10u")
                    v    = mv.read(data=g, param="10v")
                    ws   = mv.sqrt(u*u + v*v)
                    ws   = mv.grib_set_long(ws, ['paramId', 207])
                else:
                    raise ValueError("Cannot open file " + filename) 

                times = mv.valid_date(ws)
                
                for tv in zip(times, ws):
                    data.append(tv)
            
            return data
        else:
            raise ValueError("windspeed_extract_timeseries_10m only supports prefix=single")

    # extrtact the time series of a scalar variable defined on pressure levels
    def variable_extract_timeseries_lvl(self, var="t", level="500"):

        """
        Extract the time series of a scalar variable defined on pressure levels.

        Parameters
        ----------
        var : str
            The variable to extract.
        level : str
            The pressure level to extract.

        Returns
        -------
        list of tuple
            The time series and variable data.
        """

        if(self.prefix == "levels"):
            filenames = self.get_filenames()
            data = []
            for filename in filenames:
                if mv.exist(filename):
                    g   = mv.read(filename)
                    m   = mv.read(data=g, param=var, levelist=level)
                else:
                    raise ValueError("Cannot open file " + filename) 

                times = mv.valid_date(m)
                
                for tv in zip(times, m):
                    data.append(tv)
            
            return data
        else:
            raise ValueError("variable_extract_timeseries_slv only supports prefix=single")
        
    # extrtact the time series of a scalar variable defined on single level
    def variable_extract_timeseries_slv(self, var="sp"):

        """
        Extract the time series of a scalar variable defined on single levels.

        Parameters
        ----------
        var : str
            The variable to extract.

        Returns
        -------
        list of tuple
            The time series and variable data.
        """

        if(self.prefix == "single"):
            filenames = self.get_filenames()
            data = []
            for filename in filenames:
                if mv.exist(filename):
                    g   = mv.read(filename)
                    m   = mv.read(data=g, param=var)
                else:
                    raise ValueError("Cannot open file " + filename) 

                times = mv.valid_date(m)
                
                for tv in zip(times, m):
                    data.append(tv)
            
            return data
        else:
            raise ValueError("variable_extract_timeseries_slv only supports prefix=single")

    # get plotting variables 
    def def_mv_plot_settings(self, minmax=[0, 0]):

        """
        Defines the plot settings to be passed to Metview contour plots mv.plot().

        Parameters
        ----------
        minmax : list of float
            The minimum and maximum values of the variable.

        Returns
        -------
        tuple
            The view and contour settings.
        """

        cmin   = np.floor(minmax[0])
        cmax   = np.ceil(minmax[1])

        # get land-sea boundary data  
        coast = mv.mcoast(
            map_coastline_land_shade        = "on",
            map_coastline_land_shade_colour = "grey",
            map_coastline_sea_shade         = "on",
            map_coastline_sea_shade_colour  = "RGB(0.8944,0.9086,0.933)",
            map_coastline_thickness         =  2,
            map_boundaries                  = "on",
            map_boundaries_colour           = "charcoal",
            map_grid_colour                 = "charcoal",
            map_grid_longitude_increment    = 10
            )
        
        # get min max lat and lon
        [min_lat, max_lat, min_lon, max_lon] = self.get_min_max_lat_lon()

        # define the view 
        view = mv.geoview(
            map_area_definition = 'corners',
            area = [min_lat, min_lon, max_lat, max_lon],
            coastlines = coast
            )
    
        # define the contour settings
        diff_cont = mv.mcont(
            legend="on",
            contour="off",
            contour_max_level=cmax,
            contour_min_level=cmin,
            contour_level_selection_type="count",
            contour_shade="on",
            contour_shade_technique="grid_shading",
            contour_shade_colour_method="palette",
            contour_shade_palette_name="eccharts_blue_white_red_9",
            grib_scaling_of_retrieved_fields="off",
        )

        return [view, diff_cont]

    # create plots of the time series of wind speed at a given pressure level
    def windspeed_plot_timeseries_lvl(self, path2output="./snapshots_ws", level="500"):
        
        """
        Plot time series of windspeed at a given level. Data must be defined on pressure levels.

        Parameters
        ----------
        path2output : str
            The path to the output directory.
        level : str
            The pressure level to plot.
        """

        # get wind speed at 500 hPa
        ws = self.windspeed_extract_timeseries_lvl(level)  

        # get number of time samples 
        n_time_u = len(ws)
        
        # get min max value of the variable
        min_val =  1000000.0
        max_val = -1000000.0

        # get min a max value of the variable
        for t in range(n_time_u):
            min_val = min(min_val, mv.minvalue(ws[t][1]))
            max_val = max(max_val, mv.maxvalue(ws[t][1]))

        # define plot settings 
        [view, diff_cont] = self.def_mv_plot_settings(minmax=[min_val, max_val])

        # Create the folder
        os.makedirs(path2output, exist_ok=True)

        for t in range(n_time_u):
            filename = f"{path2output}/ws_{level}_{t}"
            mv.setoutput(mv.png_output(output_width=800, output_name = filename, output_font_scale=2))
            mv.plot(ws[t][1], view, diff_cont) 

    # create plots of the time series of wind speed at 10m
    def windspeed_plot_timeseries_10m(self, path2output="./snapshots_ws10"):
        
        """
        Plot time series of 10 wind speed. Data must be defined on single levels.

        Parameters
        ----------
        path2output : str
            The path to the output directory.
        var : str
            The variable to plot.
        """

        # get wind speed at 500 hPa
        ws = self.windspeed_extract_timeseries_10m()  

        # get number of time samples 
        n_time_u = len(ws)
        
        # get min max value of the variable
        min_val =  1000000.0
        max_val = -1000000.0

        # get min a max value of the variable
        for t in range(n_time_u):
            min_val = min(min_val, mv.minvalue(ws[t][1]))
            max_val = max(max_val, mv.maxvalue(ws[t][1]))

        # define plot settings 
        [view, diff_cont] = self.def_mv_plot_settings(minmax=[min_val, max_val])

        # Create the folder
        os.makedirs(path2output, exist_ok=True)

        for t in range(n_time_u):
            filename = f"{path2output}/ws10_{t}"
            mv.setoutput(mv.png_output(output_width=800, output_name = filename, output_font_scale=2))
            mv.plot(ws[t][1], view, diff_cont) 

    # create plots of the time series of given variable defined on single levels
    def variable_plot_timeseries_slv(self, path2output="./snapshots_slv", var="sp"):
        
        """
        Plot time series of variable defined on single levels.

        Parameters
        ----------
        path2output : str
            The path to the output directory.
        var : str
            The variable to plot.
        """

        # get variable
        m = self.variable_extract_timeseries_slv(var)  

        # get number of time samples 
        n_time_u = len(m)
        
        # Create the folder
        os.makedirs(path2output, exist_ok=True)

        # get min max value of the variable
        min_val =  1000000.0
        max_val = -1000000.0

        # get min a max value of the variable
        for t in range(n_time_u):
            min_val = min(min_val, mv.minvalue(m[t][1]))
            max_val = max(max_val, mv.maxvalue(m[t][1]))

        # define plot settings 
        [view, diff_cont] = self.def_mv_plot_settings(minmax=[min_val, max_val])

        for t in range(n_time_u):
            filename = f"{path2output}/{var}_{t}"
            mv.setoutput(mv.png_output(output_width=800, output_name = filename, output_font_scale=2))
            mv.plot(m[t][1], view, diff_cont) 

    # compute wind stats at a point given wind components 
    def compute_wind_stats(self, u, v, lat=0, lon=0):
        
        """
        Compute wind statistics.

        Parameters
        ----------
        point : list of float
            The latitude and longitude of the point.
        time : list of datetime
            The time series.
        hlevs : list of float
            The height levels.
        u_1d : list of float
            The u-component of the wind.
        v_1d : list of float
            The v-component of the wind.

        Returns
        -------
        DataFrame
            The wind statistics.
        """

        point = [lat, lon]
        n_time_u = len(u)
        n_time_v = len(v)

        if n_time_u != n_time_v:
            raise ValueError("u and v cannot have different lengths")
        
        ws_loc  = []
        u_loc   = []
        v_loc   = []
        dir_loc = []
        time    = []

        # get components of wind at the given point
        for t in range(n_time_u):
            u_loc.append(mv.interpolate(u[t][1],point)) 
            v_loc.append(mv.interpolate(v[t][1],point)) 
            ws_loc.append(mv.sqrt(u_loc[t]*u_loc[t] + v_loc[t]*v_loc[t]))   
            dir_loc.append((np.degrees(np.arctan2(u_loc[t], v_loc[t])) + 360 + 180) % 360)
            time.append(u[t][0])
        
        # get day numbers 
        day_numbers = [pd.to_datetime(t).day for t in time]

        # remove duplicates 
        day_numbers = list(set(day_numbers))

        # create date time object for plot x axis 
        year             = time[0].year
        month            = time[0].month
        datetime_objects = [datetime(year, month, day) for day in day_numbers]

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].plot(time, ws_loc, marker='none', linestyle='-', color='b')
        axs[0, 0].set_ylabel('Wind speed [m/s]')

        axs[0, 1].plot(time, dir_loc, marker='none', linestyle='-', color='b')
        axs[0, 1].set_ylabel('Wind Dir [deg]')

        axs[1, 0].plot(time, v_loc, marker='none', linestyle='-', color='b')
        axs[1, 0].set_ylabel('V component wind [m/s]')

        axs[1, 1].plot(time, u_loc, marker='none', linestyle='-', color='b')
        axs[1, 1].set_ylabel('U component wind [m/s]')

        for ax in axs.flat:
            ax.set_xticks(datetime_objects)
            ax.set_xticklabels([dt.strftime('%d-%m-%y') for dt in datetime_objects], rotation=45)

        plt.tight_layout()
        plt.savefig("10m_wind_tseries.png")
        plt.close()


        # Define the number of sectors (20 degrees each)
        bin_step    = 20
        num_sectors = 360 // 20
        bins        = np.linspace(0, 360, num_sectors + 1)
        maxmag      = np.max(ws_loc)

        # Calculate the histogram for wind directions
        hist, bin_edges = np.histogram(dir_loc, bins=bins)

        # Calculate the probability for each sector
        probabilities   = hist / len(dir_loc)

        # Create the wind rose plot
        fig, axs = plt.subplots(3, 2, figsize=(10, 16), subplot_kw={'projection': 'polar'})

        # find angles corresponding to middle of each sector
        angles  = np.deg2rad(bin_edges[:-1] + bin_step/2)  
        bars    = axs[0, 0].bar(angles, probabilities, width=np.deg2rad(bin_step*0.9), bottom=0.0, color=plt.cm.viridis(probabilities
        ), edgecolor='black')

        # Remove probability labels
        axs[0, 0].set_yticklabels([])

        # set the direction of the wind rose (clockwise)
        axs[0, 0].set_theta_direction(-1)

        # set the location of 0 degrees to the top (north)
        axs[0, 0].set_theta_offset(np.pi / 2.0)
        axs[0, 0].set_title(f'Wind dir. distrib. at [lat,lon]: [{point[0]},{point[1]}]')

        # Calculate the most likely magnitude, average and standard deviation for each sector
        most_likely_magnitudes   = []
        average_magnitudes       = []
        std_deviation_magnitudes = []
        max_magnitudes           = []
        min_magnitudes           = []
        for i in range(num_sectors):
            sector_mask = (dir_loc >= bins[i]) & (dir_loc < bins[i + 1])
            if np.any(sector_mask):
                most_likely_magnitudes.append(np.median(np.array(ws_loc)[sector_mask]))
                average_magnitudes.append(np.mean(np.array(ws_loc)[sector_mask]))
                std_deviation_magnitudes.append(np.std(np.array(ws_loc)[sector_mask]))
                max_magnitudes.append(np.max(np.array(ws_loc)[sector_mask]))
                min_magnitudes.append(np.min(np.array(ws_loc)[sector_mask]))
            else:
                most_likely_magnitudes.append(0)
                average_magnitudes.append(0)
                std_deviation_magnitudes.append(0)
                max_magnitudes.append(0)
                min_magnitudes.append(0)

        # Create the polar plot for most likely magnitudes
        bars = axs[0, 1].bar(angles, most_likely_magnitudes, width=np.deg2rad(bin_step*0.9), edgecolor='black')
        axs[0, 1].set_theta_direction(-1)
        axs[0, 1].set_theta_offset(np.pi / 2.0)
        axs[0, 1].set_title('Wind most likely mag [m/s]')
        axs[0, 1].set_ylim(0, maxmag)

        # Create the polar plot for wind mean 
        bars = axs[1, 0].bar(angles, average_magnitudes, width=np.deg2rad(bin_step*0.9), edgecolor='black')
        axs[1, 0].set_theta_direction(-1)
        axs[1, 0].set_theta_offset(np.pi / 2.0)
        axs[1, 0].set_title('Wind mean [m/s]')
        axs[1, 0].set_ylim(0, maxmag)

        # Create the polar plot for wind std
        bars = axs[1, 1].bar(angles, std_deviation_magnitudes, width=np.deg2rad(bin_step*0.9), edgecolor='black')
        axs[1, 1].set_theta_direction(-1)
        axs[1, 1].set_theta_offset(np.pi / 2.0)
        axs[1, 1].set_title('Wind std. deviation [m/s]')
        axs[1, 1].set_ylim(0, maxmag)

        # Create the polar plot for max wind
        bars = axs[2, 0].bar(angles, max_magnitudes, width=np.deg2rad(bin_step*0.9), edgecolor='black')
        axs[2, 0].set_theta_direction(-1)
        axs[2, 0].set_theta_offset(np.pi / 2.0)
        axs[2, 0].set_title('Wind max [m/s]')
        axs[2, 0].set_ylim(0, maxmag)

        # Create the polar plot for min wind
        bars = axs[2, 1].bar(angles, min_magnitudes, width=np.deg2rad(bin_step*0.9), edgecolor='black')
        axs[2, 1].set_theta_direction(-1)
        axs[2, 1].set_theta_offset(np.pi / 2.0)
        axs[2, 1].set_title('Wind min [m/s]')
        axs[2, 1].set_ylim(0, maxmag)

        # Save the plot to a file
        plt.savefig('10m_wind_stats.png')
        plt.close() 
    
    # compute theta time series at a point 
    def theta_extract_timeseries(self, point, hlevs, hourly_step=1):

        """
        Extract potential temperature time series.

        Parameters
        ----------
        point : list of float
            The latitude and longitude of the point.
        hlevs : list of float
            The height levels.
        hourly_step : int, optional
            The hourly step for extraction (default is 1).

        Returns
        -------
        tuple
            The time series and potential temperature data.
        """

        # check if we actually have levels
        if(self.prefix == "single"):
            raise ValueError("theta_extract_timeseries only supports prefix=levels")
        
        if(hlevs[0]==0):
            raise ValueError("Cannot interpolate to ground level from grib file, increase starting level. Ground temp will be linearly extrapolated")

        filenames_l = self.get_filenames()

        # potential temperature & time vector 
        th_1d = []
        time  = []

        # this is pretty slow, might want to speed up 
        for f in range(len(filenames_l)):
            if mv.exist(filenames_l[f]):

                # print info 
                print("Processing file " + filenames_l[f])

                # get fieldsets defined on levels 
                g     = mv.read(filenames_l[f])

                # get available times 
                times = mv.unique(mv.valid_date(g))

                # for each time extract temperature and geopotential profiles at latlon points
                for t in range(len(times)):

                    if t % hourly_step == 0:

                        print(" -> time " + str(times[t]))
                        
                        # extract temperature at time t
                        t_3d_t_pl  = g.select(shortName='t', validityDateTime=times[t])

                        # get potential temperature field (ps should be encoded in the grib)
                        th_3d_t_pl = mv.pott_p(temperature=t_3d_t_pl)

                        # extract potential temperature and geopotential at time t
                        zg_3d_pl   = g.select(shortName='z',validityDateTime=times[t]) 
                        
                        # extract potential temperature at time t
                        th_3d_t_ml = mv.ml_to_hl(th_3d_t_pl, zg_3d_pl, None, hlevs, "sea", "linear")

                        # interpolate at lat-lon location
                        th_1d.append(mv.interpolate(th_3d_t_ml, point))

                        # add time to the list 
                        time.append(times[t])
                    else:
                        continue
            else:
                raise ValueError("Cannot open file " + filenames_l[f])  

        return time, th_1d    

    # compute theta stats at a point from current dataset 
    def compute_theta_stats(self, point, time, hlevs, th_1d):

        """
        Compute potential temperature statistics.

        Parameters
        ----------
        point : list of float
            The latitude and longitude of the point.
        time : list of datetime
            The time series.
        hlevs : list of float
            The height levels.
        th_1d : list of float
            The potential temperature data.

        Returns
        -------
        DataFrame
            The potential temperature statistics.
        """

        # fit a pice-wise linear model to the potential temperature profile
        params      = []

        # Create output folder
        path2output = "theta_fit"
        os.makedirs(path2output, exist_ok=True) 

        for t in range(len(time)):

            # check if first element is None and iterate until we find a valid value
            if(th_1d[t][0]==None):
                i = 1
                while th_1d[t][i] == None:
                    i += 1

            # save new profiles into a list where None values are excluded 
            th_1d_valid = th_1d[t][i:]
            hlevs_valid = hlevs[i:]
                
            # compute some quantities to define the bounds 
            gamma_tot    = (th_1d_valid[-1] - th_1d_valid[0]) / (hlevs_valid[-1] - hlevs_valid[0])
            theta_ref    = th_1d_valid[0] + (th_1d_valid[1] - th_1d_valid[0]) / (hlevs_valid[1] - hlevs_valid[0])*(-hlevs_valid[0])
            gamma_ground = (th_1d_valid[1] - th_1d_valid[0]) / (hlevs_valid[1] - hlevs_valid[0])
            
            # set the initial guess 
            initial_guess = [theta_ref,  gamma_ground,  gamma_tot, gamma_tot*250, 500.0, 250.0]  

            # Fit the model to the data
            lower_bounds =  [theta_ref-1e-5, gamma_ground-0.01, gamma_tot - 0.01,   gamma_tot*250,  0.0,    249.9999]
            upper_bounds =  [theta_ref+1e-5, gamma_ground+0.01, gamma_tot + 0.01,   10.0,           5000.0, 250.0001]


            params_t, params_covariance_t = curve_fit(
                self.piecewise_linear_potential_theta,
                    hlevs_valid,
                    th_1d_valid,
                    p0=initial_guess,
                    bounds=(lower_bounds, upper_bounds)
            )
            params.append(params_t)
            # Print the fitted parameters
            print("Fitted parameters:", params[t])

            # Plot the data and the fitted model
            plt.scatter(th_1d_valid, hlevs_valid, label='Data')
            plt.plot(self.piecewise_linear_potential_theta(np.array(hlevs), *params[t]), hlevs, color='red', label='Fitted model')
            plt.ylabel('Height (m)')
            plt.xlabel('Temperature (°C)')
            plt.legend()
            plt.title('Vertical Profile of Temperature')
            
            # Save the plot to a file
            filename    = f"{path2output}/{t}"
            plt.savefig(f'{filename}.png')
            plt.close() 

        # Convert params to DataFrame
        params_array = np.array(params)
        color_levels = 'blue'
        param_names  = ['theta_ref', 'gamma_abl', 'gamma', 'dtheta_inv', 'z_inv']
        param_lgnd   = [r'$\theta_{\mathrm{ref}}$', r'$\gamma_{\mathrm{ABL}}$', r'$\gamma$', r'$\Delta\theta$', r'$H$']
        df_params    = pd.DataFrame(params_array[:,:-1], columns=param_names)

        # Enable LaTeX rendering
        plt.rc('text', usetex=True)
        plt.rc('font', family='Times New Roman')

        # Create pairplot
        g = sns.PairGrid(df_params, diag_sharey=False, corner=True)

        # Define a function to add vertical bars on the diagonal
        def diag_mean(data, **kwargs):
            median = data.median()
            mean = data.mean()
            plt.axvline(median, color='black', linestyle='--')
            plt.axvline(mean, color='black', linestyle=':')
            sns.kdeplot(data, **kwargs)

        # Define a function to add vertical and horizontal bars in the subdiagonal plots
        def off_diag_mean(x, y, **kwargs):
            median_x = x.median()
            median_y = y.median()
            mean_x = x.mean()
            mean_y = y.mean()
            plt.axvline(median_x, color='black', linestyle='--')
            plt.axhline(median_y, color='black', linestyle='--')
            plt.axvline(mean_x, color='black', linestyle=':')
            plt.axhline(mean_y, color='black', linestyle=':')
            sns.kdeplot(x=x, y=y, **kwargs)

        # Map the functions to the grid
        g.map_diag(diag_mean, color='b', alpha=1)
        g.map_offdiag(off_diag_mean, color=color_levels)

        # Update axis labels with LaTeX notation
        for i, ax in enumerate(g.axes.flatten()):
            if ax is not None:
                ax.set_xlabel(f"${param_lgnd[i % len(param_lgnd)]}$", fontsize=15)
                ax.set_ylabel(f"${param_lgnd[i // len(param_lgnd)]}$", fontsize=15)

        # Add title in the top right of the diagram
        time_interval = f"{time[0]} to {time[-1]}"
        title_text    = f"Potential temperature individual and joint statistics\nat latitude: {point[0]} deg, longitude: {point[1]} deg\ntime interval: {time_interval}\nDashed line: median, dotted line: mean"

        # Calculate the position for the text
        plt.subplots_adjust(top=0.9)  # Adjust the top to make space for the title
        g.figure.text(0.95, 0.95, title_text, ha='right', va='top', fontsize=20)

        # Show plot
        plt.show()

        # Save the plot to a file
        plt.savefig('theta_stats.png')

        return df_params

    # provides a picewise-linear potential temperature models given lapse rate, surface temp, inversion height/strength and free atm lapse rate 
    def piecewise_linear_potential_theta(self, z, theta_ref, gamma_abl, gamma, dtheta_inv, z_inv, delta_inv):

        """
        Compute the piecewise-linear potential temperature model.

        Parameters
        ----------
        z : float
            The height.
        theta_ref : float
            The reference potential temperature.
        gamma_abl : float
            The adiabatic lapse rate in the atmospheric boundary layer.
        gamma : float
            The free atmospheric lapse rate.
        dtheta_inv : float
            The potential temperature inversion strength.
        z_inv : float
            The height of the inversion.
        delta_inv : float
            The inversion width.

        Returns
        -------
        float
            The potential temperature profile on z.
        """

        half_inv  = delta_inv / 2.0
        z_inv_lo  = z_inv - half_inv
        z_inv_hi  = z_inv + half_inv
        gamma_inv = dtheta_inv / delta_inv

        f = np.zeros(len(z))
        for i in range(len(z)):
            if(z[i] < z_inv_lo):
                f[i] = theta_ref + gamma_abl * z[i]
            elif(z[i] < z_inv_hi and z[i] >= z_inv_lo):
                f[i] = theta_ref + gamma_abl * z_inv_lo + gamma_inv * (z[i] - z_inv_lo)
            else:
                f[i] = theta_ref + gamma_abl * z_inv_lo + dtheta_inv + gamma * (z[i] - z_inv_hi)
            
        return(f)
    