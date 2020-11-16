#!/usr/bin/env python
# coding: utf-8

from joblib import Parallel, delayed
import math
import timeit
import numpy as np
import pandas as pd
import sys
import os


def data_augmentation_one_year(train_date, covariate_set, spatial_range, path):
    """Create a dataframe of all variables in covariate_set within the required
    saptial and temporal range (within one year)

    Args:
        train_date: a tuple as (start_date, end_date)
        covariate_set: a list of climate variables
        spatial_range: a dataframe with required latitude and longtitude
        path: the rootpath for reading the raw data of each climate variable

    Returns:
        A multiindex dataframe with requested cliamte variables
    """
    # idx = pd.IndexSlice
    # date_index=pd.date_range(start=train_date[0],end=train_date[1])
    cov = covariate_set[0]
    file_name = path + cov + '.' + str(train_date[0].year) + '.h5'
    cov_file = pd.read_hdf(file_name)
    year = train_date[0].year
    if train_date[0] == pd.Timestamp(year, 1, 1) and train_date[1] == pd.Timestamp(year, 12, 31):
        cov_temp = cov_file.reset_index()
        df = cov_temp.merge(spatial_range, on=['lat', 'lon'])
    else:
        date_index = pd.date_range(start=train_date[0], end=train_date[1])
        date_index = pd.DataFrame(date_index, columns=['start_date'])
        cov_temp = cov_file.reset_index()
        cov_temp = date_index.merge(cov_temp, on=['start_date'])
        df = cov_temp.merge(spatial_range, on=['lat', 'lon'])
    df = df.rename(columns={df.columns[-1]: cov})
    if len(cov) > 1:
        for cov in covariate_set[1:]:
            file_name = path + cov + '.' + str(train_date[0].year) + '.h5'
            cov_temp = pd.read_hdf(file_name)
            # cov_temp = cov_file.to_frame()
            df = df.merge(cov_temp, on=['lat', 'lon', 'start_date'])
            df = df.rename(columns={df.columns[-1]: cov})
    df.set_index(['lat', 'lon', 'start_date'], inplace=True)
    return df


class DataLoader(object):
    """
    Download requested cliamte variables and created dataframe for target variable and covariates
    """
    def __init__(self, args):
        self.path = args.path
        self.path_save = args.path_save
        self.target_variable = args.target
        self.target_lat = args.target_lat
        self.target_lon = args.target_lon
        self.target_us_all = args.target_us_all
        if self.target_us_all is True:
            self.target_res = args.target_res

        self.covariate_set_us = args.covariates_us
        self.lat_range_us = args.lat_range_us
        self.lon_range_us = args.lon_range_us

        self.covariate_set_global = args.covariates_global
        self.lat_range_global = args.lat_range_global
        self.lon_range_global = args.lon_range_global

        self.ocean_covariate_set = args.covariates_sea
        self.lat_range_sea = args.lat_range_sea
        self.lon_range_sea = args.lon_range_sea
        self.pacific_atlantic = args.pacific_atlantic
        self.spatial_set = args.spatial_set
        self.temporal_set = args.temporal_set

        self.train_date_start = args.train_start_date
        self.date_end = args.end_date
        self.past_ndays = args.past_ndays
        # self.future_mdays = args.future_mdays
        self.past_kyears = args.past_kyears
        # self.train_size = args.train_range  # days
        # self.validation_size = args.val_range  # days
        # self.stride = args.stride
        self.save_target = args.save_target
        self.save_cov = args.save_cov
        self.shift_days = args.shift_days
        self.forecast_range = args.forecast_range
        self.operation = args.operation  # or "mean"
        # a dictionary with file names for all temporal/saptial climate variable
        self.variables_file = {'mei': 'mei.h5',
                               'mjo_phase': 'mjo_phase.h5',
                               'mjo_amplitude': 'mjo_amplitude.h5',
                               'elevation': 'elevation.h5',
                               'region': 'climate_regions.h5',
                               'nao': 'nao.h5',
                               'nino3': 'nino3.h5',
                               'nino4': 'nino4.h5',
                               'nino1+2': 'nino1+2.h5',
                               'nino3.4': 'nino3.4.h5'}

        # self._check_params() # need to add a function to check the input arguments

# Overall procedure
# (1) get the target variables

    def get_target_data(self, variable_id, target_lat, target_lon, train_date_start, target_end_date, resolution, path):
        """Get the target variables with the required lat, lon, date

        Args:
            variable_id: a string of target variable ('tmp2m' or 'precip')
            target_lat, target_lon: a list of the latitude/longtitude range
            train_date_start, target_end_date: Timestamp represent the start and end date for target variable
            resolution: an int represent the resolution (2)
            path: the rootpath for reading the raw data

        Returns:
            A multiindex dataframe containing target variable
        """
        if type(target_lat) is list and type(target_lon) is list:
            # this part need to be improved
            if self.target_us_all is True:
                # subsampling to get the corresponding resolution
                spatial_map = self.get_target_map(resolution, path)
            else:
                spatial_map = self.remove_masked_data('us', target_lat, target_lon, path)
            target = self.get_covariates_data_parallel_updated(train_date_start, target_end_date, [variable_id], spatial_map, path)
            return target
        else:
            target_lat = self.find_the_cloest_value(target_lat)
            target_lon = self.find_the_cloest_value(target_lon)
            if train_date_start.year == target_end_date.year:
                file_name = path + variable_id + '.' + str(train_date_start.year) + '.h5'
                target_file = pd.read_hdf(file_name)
                date_index = pd.date_range(start=train_date_start, end=target_end_date)
                target = target_file.loc[target_lat, target_lon, date_index]
                target = target.to_frame()
                target = target.rename(columns={target.columns[-1]: variable_id})
            else:
                # start year
                file_name = path + variable_id + '.' + str(train_date_start.year) + '.h5'
                target_file = pd.read_hdf(file_name)
                date_index = pd.date_range(start=train_date_start, end=pd.Timestamp(train_date_start.year, 12, 31))
                target_start = target_file.loc[target_lat, target_lon, date_index]
                target_start = target_start.to_frame()
                # end year
                file_name = path + variable_id + '.' + str(target_end_date.year) + '.h5'
                target_file = pd.read_hdf(file_name)
                date_index = pd.date_range(start=pd.Timestamp(target_end_date.year, 1, 1), end=target_end_date)
                target_end = target_file.loc[target_lat, target_lon, date_index]
                target_end = target_end.to_frame()
                if target_end_date.year - train_date_start.year > 1:
                    for year in range(train_date_start.year + 1, target_end_date.year):
                        file_name = path + variable_id + '.' + str(year) + '.h5'
                        target_file = pd.read_hdf(file_name)
                        date_index = pd.date_range(start=pd.Timestamp(year, 1, 1), end=pd.Timestamp(year, 12, 31))
                        target_file = target_file.loc[target_lat, target_lon, date_index]
                        target_temp = target_file.to_frame()
                        target_start = target_start.append(target_temp)
                target = target_start.append(target_end)
                target = target.rename(columns={target.columns[-1]: variable_id})
            return target

    def get_target_map(self, resolution, path):
        '''Get a DataFrame with spatial range of a target variable

        Args:
            resolution: an int represent the resolution (2)
            path: the rootpath for reading the raw data
        Returns:
            a DataFrame with spatial range (lat and lon) of a target variable
        '''
        if resolution == 2:
            # subsample the map with the resolution as 2x2 for target variables
            spatial_map = pd.read_hdf(path + 'target_map_2.h5')
            return spatial_map
        else:
            print('the spatial map for resolution as {} is not available'.format(resolution))

    def shift_target(self, target_df, target_lat, target_lon, target_id, shift_days, forecast_range, operation):
        """Shift the target variable add compute summation or average values for the required forecast_range
        Args:
            target_df: a DataFrame with target variable
            target_lat, target_lon: a list of the latitude/longtitude range
            target_id: a string of target variable ('tmp2m' or 'precip')
            shift_days: number of days by which the target variable should be shifted forward (14 for 2 weeks and 28 for 4 weeks)
            forecast_range: number of days by which the target variable should be aggreated (14 for 2 weeks and 28 for 4 weeks)
            operation: the operation for aggreating the target variable over forecasting range ('sum', 'mean', or 'median')
        Returns:
            A DataFrame with shifted and aggreated target variables
            (e.g. if shift_days = 14 and forecast_range =14, it returns the target variable needed for forecasting week 3 and 4)
        """
        if type(target_lat) is list and type(target_lon) is list:
            idx = pd.IndexSlice
            target_df = target_df.reset_index()
            target_df = target_df.set_index(['start_date', 'lat', 'lon'])
            target_df = target_df.unstack(['lat', 'lon'])
            date_list = target_df.index.get_level_values('start_date')
            num_date = len(date_list) - forecast_range + 1
            if operation == "sum":
                for date in date_list[:num_date]:
                    date_start = date
                    date_end = date + pd.DateOffset(forecast_range - 1)
                    temp = target_df.loc[date_start:date_end].values
                    target_df.loc[date_start] = np.sum(temp, axis=0)
            elif operation == "mean":
                for date in date_list[:num_date]:
                    date_start = date
                    date_end = date + pd.DateOffset(forecast_range - 1)
                    temp = target_df.loc[date_start: date_end].values
                    target_df.loc[date_start] = np.mean(temp, axis=0)
            elif operation == "median":
                for date in date_list[:num_date]:
                    date_start = date
                    date_end = date + pd.DateOffset(forecast_range - 1)
                    temp = target_df.loc[date_start:date_end].values
                    target_df.loc[date_start] = np.median(temp, axis=0)
            target_df.loc[date_list[num_date]:date_list[-1]] = float('nan')
            target_df = target_df.shift(-shift_days)
            target_df = target_df.dropna()
            target_df = target_df.stack(['lat', 'lon'])
            target_df = target_df.reset_index()
            target_df = target_df.set_index(['lat', 'lon', 'start_date'])
            target_df = target_df.sort_index()
            return target_df
        else:
            idx = pd.IndexSlice
            date_list = target_df.index.get_level_values('start_date')
            num_date = len(date_list) - forecast_range + 1
            if operation == "sum":
                for date in date_list[:num_date]:
                    date_start = date
                    date_end = date + pd.DateOffset(forecast_range - 1)
                    temp = target_df.loc[idx[target_lat, target_lon, date_start:date_end], :].values
                    target_df.loc[idx[target_lat, target_lon, date_start], :] = sum(temp)
            elif operation == "mean":
                for date in date_list[:num_date]:
                    date_start = date
                    date_end = date + pd.DateOffset(forecast_range - 1)
                    temp = target_df.loc[idx[target_lat, target_lon, date_start:date_end], :].values
                    target_df.loc[idx[target_lat, target_lon, date_start], :] = np.mean(temp)
            elif operation == "median":
                for date in date_list[:num_date]:
                    date_start = date
                    date_end = date + pd.DateOffset(forecast_range - 1)
                    temp = target_df.loc[idx[target_lat, target_lon, date_start:date_end], :].values
                    target_df.loc[idx[target_lat, target_lon, date_start], :] = np.median(temp)
            target_df.loc[idx[target_lat, target_lon, date_list[num_date]:date_list[-1]], :] = float('nan')
            target_df = target_df[target_id].shift(-shift_days)
            target_df = target_df.dropna()
            return target_df.to_frame()

    def date_adapt_target(self, train_date_start, test_end_date, shift_days, forecast_range):
        '''Returns the date range to obtain to compute the target variable over the given temporal range
        e.g. need (shift_days + forecast_range) more days compared to the end date
        Args:
            train_date_start, target_end_date: Timestamp represent the start and end date for target variable
            shift_days: number of days by which the target variable should be shifted forward (14 for 2 weeks and 28 for 4 weeks)
            forecast_range: number of days by which the target variable should be aggreated (14 for 2 weeks and 28 for 4 weeks)
        Returns:
            data_start_date, data_end_date: timestamps represent the temporal boundary to obtain
        '''
        data_start_date = pd.Timestamp(train_date_start)
        data_end_date = pd.Timestamp(test_end_date)
        data_end_date = data_end_date + pd.DateOffset(days=shift_days + forecast_range - 1)
        return data_start_date, data_end_date

# (2) common functions

    def find_the_cloest_value(self, lat):
        """Find the cloest lat/lon for the resolution as 0.5 x 0.5 with the given lat/lon

        Args:
            lat: a value represent either lat or lon

        Returns:
            a value with in the lat/lon grid given the resoluion as 0.5 x 0.5
        """
        if lat > math.floor(lat) + 0.5:
            lat_min = math.floor(lat) + 0.75
        else:
            lat_min = math.floor(lat) + 0.25
        return lat_min

    def get_spatial_range(self, lat_range, lon_range):
        """Get a range of latitude and longtitude given the boundary

        Args:
            lat_range/lon_range: a list of the boundary of latitude and longtitude
        Returns:
            lat_index, lon_index: the range of latitude and longtitude given the boundary
        """
        # get latitude range
        lat_min = self.find_the_cloest_value(min(lat_range))
        lat_max = self.find_the_cloest_value(max(lat_range))
        lat_index = np.arange(lat_min, lat_max + 0.5, 0.5)
        # get longtitude range
        if lon_range[0] <= lon_range[1]:
            lon_min = self.find_the_cloest_value(lon_range[0])
            lon_max = self.find_the_cloest_value(lon_range[1])
            lon_index = np.arange(lon_min, lon_max + 0.5, 0.5)
        else:
            lon_min = self.find_the_cloest_value(lon_range[1])
            lon_max = self.find_the_cloest_value(lon_range[0])
            lon_index = np.concatenate((np.arange(0.25, lon_min + 0.5, 0.5), np.arange(lon_max, 360.25, 0.5)), axis=None)
        return lat_index, lon_index

    def remove_masked_data(self, mask_area, lat_range, lon_range, path):
        """Get the spatial range of a sepecif area within the given the requested lat/lon range

        Args:
            mask_area: the name of the mask area ('us', 'ocean' or 'global')
            (the mask area is the area left - may need to change the name)
            lat_range/lon_range: a list of the boundary of latitude and longtitude
            path: the rootpath for reading the raw data

        Returns:
            A dataframe with latitude and longtitude of a sepecif area within the given the requested lat/lon range
        """
        lat_index, lon_index = self.get_spatial_range(lat_range, lon_range)
        lon, lat = np.meshgrid(lon_index, lat_index)
        # the given block range
        spatial_range = pd.DataFrame({'lat': lat.flatten(), 'lon': lon.flatten()})
        if mask_area == 'us':
            mask_df = pd.read_hdf(path + 'us_mask.h5')
            # the range after remove non-land area
            spatial_range = pd.merge(spatial_range, mask_df, on=['lat', 'lon'], how='inner')
        elif mask_area == 'ocean':
            mask_df = pd.read_hdf(path + 'sst_mask.h5')
            spatial_range = pd.merge(spatial_range, mask_df, on=['lat', 'lon'], how='inner')
        else:
            print('no mask is applied')
        return spatial_range

    def get_data_file(self, variable_id, path):
        """Get the correspoding data file from each variable's name (only for temporal and saptial variables)
        Args:
            variable_id: a string of a spatial or temporal variable ('mei', 'elevation', etc)
        Returns:
            A dataframe of the required cliamte variable
            path: the rootpath for reading the raw data
        """
        file_name = path + self.variables_file[variable_id]
        return pd.read_hdf(file_name)

# (3) functions for geting the cov variables

    def get_covariates_data_parallel_updated(self, train_date_start, target_end_date, covariate_set, spatial_range, path):
        """Get the covariate with the required range
        Args:
            train_date_start, target_end_date: Timestamp represent the start and end date for covariates
            covariate_set: a list of climate variables
            spatial_range: a dataframe with required latitude and longtitude
            path: the rootpath for reading the raw data of each climate variable

        Returns:
            A multiindex dataframe with requested cliamte variables

        """
        if train_date_start.year == target_end_date.year:
            train_date_index = (train_date_start, target_end_date)
            df = data_augmentation_one_year(train_date_index, covariate_set, spatial_range, path)
        else:
            train_date_index = [(train_date_start, pd.Timestamp(train_date_start.year, 12, 31))]
            if (target_end_date.year - train_date_start.year) > 1:
                for year in range(train_date_start.year + 1, target_end_date.year):
                    train_date_index.append((pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)))
            train_date_index.append((pd.Timestamp(target_end_date.year, 1, 1), target_end_date))
            results = Parallel(n_jobs=8)(delayed(data_augmentation_one_year)(train_date_temp, covariate_set, spatial_range, path) for train_date_temp in train_date_index)
            df = pd.concat(results)
        df.sort_index(ascending=True, inplace=True)
        return df

    def get_temporal_subset(self, data, start_date, end_date):
        """Get the subset data for each temporal covariate based on the required temporal range
        Args:
            data: A DataFrame with a temporal climate variable
            start_date/end_date: the timestamp of the start and end date for the required temporal range
        Returns:
            A DataFrame of a temporal varialbe with the required temporal range
        """
        date_index = pd.date_range(start=start_date, end=end_date)
        date_index = pd.DataFrame(date_index, columns=['start_date'])
        data_temp = data.reset_index()
        data_subset = date_index.merge(data_temp, on=['start_date'])
        return data_subset

    def get_spatial_subset(self, data, lat_range, lon_range):
        """Get the subset data for each covariate based on the required spatial range
        Args:
            data: A DataFrame with a temporal climate variable
            lat_range/lon_range: lat_range/lon_range: a list of the boundary of latitude and longtitude
        Returns:
        A DataFrame of a spatial varialbe with the required spatial range
        """
        lat_min = self.find_the_cloest_value(min(lat_range))
        lat_max = self.find_the_cloest_value(max(lat_range))
        lat_index = np.arange(lat_min, lat_max + 0.5, 0.5)
        # get longtitude range
        if lon_range[0] <= lon_range[1]:
            lon_min = self.find_the_cloest_value(lon_range[0])
            lon_max = self.find_the_cloest_value(lon_range[1])
            lon_index = np.arange(lon_min, lon_max + 0.5, 0.5)
        else:
            lon_min = self.find_the_cloest_value(lon_range[1])
            lon_max = self.find_the_cloest_value(lon_range[0])
            lon_index = chain(np.arange(0, lon_min + 0.5, 0.5), np.arange(lon_max, 359.75))

        return data.loc[lat_index, lon_index]  # the way to slice the data can be improved

    def split_pacific_atlantic(self, rootpath, covariates_sea):
        '''Split the covariates over sea to north pacific and north atlantic ocean
        Args:
            rootpath: the rootpath for reading the mask file
            covariates_sea: a DataFrame with all covariates over sea
        Returns:
            covariates_sea_pacific, covariates_sea_atlantic: DataFrames with the covariates over north pacific ocean and north atlantic ocean
        '''
        atlantic_mask = pd.read_hdf(rootpath + 'atlantic_mask.h5')
        pacific_mask = pd.read_hdf(rootpath + 'pacific_mask.h5')
        covariates_sea_temp = covariates_sea.reset_index()
        covariates_sea_pacific = pacific_mask.merge(covariates_sea_temp, on=['lat', 'lon'], how='left')
        covariates_sea_pacific.set_index(['lat', 'lon', 'start_date'], inplace=True)
        covariates_sea_pacific.sort_index(ascending=True, inplace=True)
        covariates_sea_atlantic = atlantic_mask.merge(covariates_sea_temp, on=['lat', 'lon'], how='left')
        covariates_sea_atlantic.set_index(['lat', 'lon', 'start_date'], inplace=True)
        covariates_sea_atlantic.sort_index(ascending=True, inplace=True)
        return covariates_sea_pacific, covariates_sea_atlantic

    def date_adapt(self, train_date_start, test_end_date, past_kyears, past_ndays):
        '''Returns the date range to obtain to create the feature set for all covariates
        Args:
            train_date_start, target_end_date: Timestamp represent the start and end date for covariates set
            past_kyears: number of years by which the covariates should be shifted backward
            past_ndays: number of days by which the covariates should be shifted backward
        Returns:
            data_start_date, data_end_date: the start and end date range to obtain to create the feature set for all covariates
        '''
        data_start_date = pd.Timestamp(train_date_start)
        data_start_date = data_start_date - pd.DateOffset(years=past_kyears, days=past_ndays)
        data_end_date = pd.Timestamp(test_end_date)
        # data_end_date = data_end_date + pd.DateOffset(days=shift_days+forecast_range-1)
        return data_start_date, data_end_date

    def create_date_data(self, covariates_set, start_date, end_date, path):
        """Create a dataframe for all temporal covariates
        Args:
            covariates_set: a list of temporal climate variables
            start_date, end_date: timestamps represent the temporal boundary to obtain
            path: the rootpath for reading the raw data
        Returns:
            A DataFrame with temporal covariates
        """
        cov_id = covariates_set[0]
        cov_file = self.get_data_file(cov_id, path)
        cov = self.get_temporal_subset(cov_file, start_date, end_date)
        cov = cov.rename(columns={cov.columns[-1]: cov_id})
        if len(covariates_set) > 1:
            covariates_set = covariates_set[1:]
            for cov_id in covariates_set:
                cov_file = self.get_data_file(cov_id, path)
                cov_subset = self.get_temporal_subset(cov_file, start_date, end_date)
                cov_temp = pd.DataFrame(cov_subset)
                cov_temp = cov_temp.rename(columns={cov_temp.columns[-1]: cov_id})
                cov = pd.merge(cov, cov_temp, on=["start_date"], how="left")
        cov = cov.set_index(['start_date'])  # convert date information to index
        return cov

    def create_lat_lon_data(self, covariates_set, lat_range, lon_range, path):
        """Create a dataframe for all spatial covariates
        Args:
            covariates_set: a list of spatial climate variables
            lat_range/lon_range: a list of the boundary of latitude and longtitude
            path: the rootpath for reading the raw data
        Returns:
            A DataFrame with spatial covariates
        """
        cov_id = covariates_set[0]
        cov_file = self.get_data_file(cov_id, path)
        cov_subset = self.get_spatial_subset(cov_file, lat_range, lon_range)
        cov = pd.DataFrame(cov_subset)
        cov.reset_index(inplace=True)
        cov = cov.rename(columns={cov.columns[-1]: cov_id})
        if len(covariates_set) > 1:
            covariates_set = covariates_set[1:]
            for cov_id in covariates_set:
                cov_file = self.get_data_file(cov_id, path)
                cov_subset = self.get_spatial_subset(cov_file, lat_range, lon_range)
                cov_temp = pd.DataFrame(cov_subset)
                cov_temp = cov_temp.rename(columns={cov_temp.columns[-1]: cov_id})
                cov = pd.merge(cov, cov_temp, on=["lat", "lon"], how="left")
        return cov
# (4) Combination of all operations

    def data_download_target(self):
        '''Load the target variable based on user's query
        Returns:
            A multiindex DataFrame with target variable
        '''
        # add past years and days to the dataset
        print('compute the real date range considering shift and forecast range')
        train_date_start, target_end_date = self.date_adapt_target(self.train_date_start, self.date_end, self.shift_days, self.forecast_range)
        # get target variable
        print('obtain target data')
        target = self.get_target_data(self.target_variable, self.target_lat, self.target_lon, train_date_start, target_end_date, self.target_res, self.path)
#        print('shift and compute average for target')
        if self.shift_days > 0 or self.forecast_range > 0:
            print('shift and compute average for target')
            target = self.shift_target(target, self.target_lat, self.target_lon, self.target_variable, self.shift_days, self.forecast_range, self.operation)

        if self.save_target is True:
            target.to_hdf(self.path_save + 'target.h5', key='target', mode='w')
            print('target data saved')
        return target

    def data_download_cov(self):
        '''Load the covariates based on user's query
        Returns:
            Covariates_us, covariates_sea, covariates_global: multiindex DataFrames of spatial covariates over us mainland, ocean, and global range
            Spatial_covariates, temporal_covariates: DataFrames with spatial and temporal variables
        '''
        # add past years and days to the dataset
        train_date_start, target_end_date = self.date_adapt(self.train_date_start, self.date_end, self.past_kyears, self.past_ndays)
        # get spatial-temporal covariates on land
        print('load data for us continent')
        tic = timeit.default_timer()
        if len(self.covariate_set_us) > 0:
            us_land_spatial_range = self.remove_masked_data('us', self.lat_range_us, self.lon_range_us, self.path)
            covariates_us = self.get_covariates_data_parallel_updated(train_date_start, target_end_date, self.covariate_set_us, us_land_spatial_range, self.path)
            if self.save_cov is True:
                covariates_us.to_hdf(self.path_save + 'covariates_us.h5', key='covariates_us', mode='w')
        else:
            covariates_us = None
        toc = timeit.default_timer()
        print(toc - tic)
        print('load data for global scale')
        tic = timeit.default_timer()
        if len(self.covariate_set_global) > 0:
            spatial_range_global = self.remove_masked_data('all', self.lat_range_global, self.lon_range_global, self.path)
            covariates_global = self.get_covariates_data_parallel_updated(train_date_start, target_end_date, self.covariate_set_global, spatial_range_global, self.path)
            if self.save_cov is True:
                covariates_global.to_hdf(self.path_save + 'covariates_global.h5', key='covariates_global', mode='w')
        else:
            covariates_global = None
        toc = timeit.default_timer()
        print(toc - tic)

        # get spatial-temporal covariates on sea
        print('load data for ocean')
        if len(self.ocean_covariate_set) > 0:
            ocean_spatial_range = self.remove_masked_data('ocean', self.lat_range_sea, self.lon_range_sea, self.path)
            covariates_sea = self.get_covariates_data_parallel_updated(train_date_start, target_end_date, self.ocean_covariate_set, ocean_spatial_range, self.path)
            # covariates_sea is not sorted, add following
            covariates_sea.sort_index(ascending=True, inplace=True)
            if self.pacific_atlantic is True:
                covariates_sea_pacific, covariates_sea_atlantic = self.split_pacific_atlantic(self.path, covariates_sea)
                if self.save_cov is True:
                    covariates_sea_pacific.to_hdf(self.path_save + 'covariates_pacific.h5', key='covariates_pacific', mode='w')
                    covariates_sea_atlantic.to_hdf(self.path_save + 'covariates_atlantic.h5', key='covariates_atlantic', mode='w')
            else:
                if self.save_cov is True:
                    covariates_sea.to_hdf(self.path_save + 'covariates_sea.h5', key='covariates_sea', mode='w')
        else:
            covariates_sea = None

        # get spatial covariates
        print('load spatial data')
        if len(self.spatial_set) > 0:
            spatial_covariates = self.create_lat_lon_data(self.spatial_set, self.lat_range_global, self.lon_range_global, self.path)
            # spatial_covariates is not multiindexed, add following
            spatial_covariates.set_index(['lat', 'lon'], inplace=True)
            if self.save_cov is True:
                spatial_covariates.to_hdf(self.path_save + 'spatial_covariates.h5', key='spatial_covariates', mode='w')
        else:
            spatial_covariates = None

        # get temporal covariates
        print('load temporal data')
        if len(self.temporal_set) > 0:
            temporal_covariates = self.create_date_data(self.temporal_set, train_date_start, target_end_date, self.path)
            if self.save_cov is True:
                temporal_covariates.to_hdf(self.path_save + 'temporal_covariates.h5', key='temporal_covariates', mode='w')
        else:
            temporal_covariates = None
        return covariates_us, covariates_sea, covariates_global, spatial_covariates, temporal_covariates
