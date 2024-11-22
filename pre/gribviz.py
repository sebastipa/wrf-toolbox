import metview as mv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from   datetime import datetime

class gribdata:

    # dummy initialization
    start_day     = 1
    end_day       = 1
    start_month   = 1
    end_month     = 1
    year          = 2022
    path          = "/storage/wrf/nobackup/munters/ERA5"
    prefix        = "levels"
    extension     = "grib"
    
    # constants 
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # class constructor
    def __init__(self, day_range =[1, 1], 
                 month_range     =[1, 1], 
                 year            =2022, 
                 prefix          ="levels", 
                 extension       = "grib", 
                 path            ="/storage/wrf/nobackup/munters/ERA5/"
                 ):
        
        # do some checks
        if day_range[0] < 1 or day_range[0] > 31:
            raise ValueError("Invalid start day")
        if day_range[1] < 1 or day_range[1] > 31:
            raise ValueError("Invalid end day")
        if month_range[0] < 1 or month_range[0] > 12:
            raise ValueError("Invalid start month")
        if month_range[1] < 1 or month_range[1] > 12:
            raise ValueError("Invalid end month")
        if month_range[0] > month_range[1]:
            raise ValueError("Start month is later than end month") 
        
        # assign values to class variables
        self.start_day   = day_range[0]
        self.end_day     = day_range[1]
        self.start_month = month_range[0]
        self.end_month   = month_range[1]
        self.year        = year
        self.path        = path
        self.prefix      = prefix
        self.extension   = extension	
        return

    # return an array of filenames ordered by date 
    def get_filenames(self):
        filenames = []
        day       = self.start_day
        month     = self.start_month
        year      = self.year

        while month <= self.end_month:
            while day <= self.days_in_month[month - 1]:
                filenames.append(self.path + "/" + str(self.year) + "/" + self.prefix + "_" + str(year) + "_" + str(month).zfill(2) + "_" + str(day).zfill(2) + ".grib")
                if (day == self.end_day and month == self.end_month):
                    break
                day += 1
            month += 1
            day = 1	
            
        return filenames  
    
    # inspect the data in the grib files
    def inspect(self):
        filenames = self.get_filenames()
        fs        = mv.read(filenames[0])
        df        = fs.ls(filter={"step":0, "time":0}, extra_keys=['name', 'units'])
        print(df.to_markdown())
        return

    # compute min max lat and lon 
    def get_min_max_lat_lon(self, verbose=False):

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

    # compute wind stats at a point 
    def compute_wind_stats(self, u, v, lat=0, lon=0):
        
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
    
    # fit a piece-wise linear temperature model at a point to find 
    # lapse rates, surface temp, inversion height/strength and lapse rate aloft
    def piecewise_linear_potential_theta(z, theta_ref, gamma_abl, gamma, dtheta_inv, z_inv):

        f = np.zeros(len(z))
        for i in range(len(z)):
            if(z[i] < z_inv):
                f[i] = theta_ref + gamma_abl * z[i]
            elif z[i] == z_inv:
                f[i] = theta_ref + gamma_abl * z[i] + dtheta_inv
            else:
                f[i] = theta_ref + gamma_abl * z_inv + dtheta_inv + gamma * (z[i] - z_inv)
            
        return(f)
    