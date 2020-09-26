#!/usr/bin/env python
# coding: utf-8

from joblib import Parallel, delayed
import math
import timeit
import numpy as np
import pandas as pd


def data_augmentation_one_year(train_date, covariate_set, spatial_range, path):
    """
    get the covariate for one year
    """
    # idx = pd.IndexSlice
    # date_index=pd.date_range(start=train_date[0],end=train_date[1])
    cov = covariate_set[0]
    # path_name='../../../../project/banerjee-00/S2S_dataset/data/'
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
#        cov_temp=cov_file.to_frame()
#        cov_temp=cov_temp.loc[idx[:,:,date_index],:]
#        df=cov_temp.reset_index().merge(spatial_range,on=['lat','lon']).set_index(['lat','lon','start_date'])
    df = df.rename(columns={df.columns[-1]: cov})
    if len(cov) > 1:
        for cov in covariate_set[1:]:
            # path_name='../../../../project/banerjee-00/S2S_dataset/data/'
            file_name = path + cov + '.' + str(train_date[0].year) + '.h5'
            cov_temp = pd.read_hdf(file_name)
            # cov_temp = cov_file.to_frame()
            df = df.merge(cov_temp, on=['lat', 'lon', 'start_date'])
            df = df.rename(columns={df.columns[-1]: cov})
    df.set_index(['lat', 'lon', 'start_date'], inplace=True)
    return df


class DataLoader(object):
    """
    Download data and split it to multiple training and validation sets
    """
    def __init__(self, args):
        self.path = args.path
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
#        self.test_date_start= args.test_start_date
        self.date_end = args.end_date
        self.past_ndays = args.past_ndays
        self.future_mdays = args.future_mdays
        self.past_kyears = args.past_kyears
        # SH: Shall we convert to years? Leap years are kinda problemic.
        self.train_size = args.train_range  # days
        self.validation_size = args.val_range  # days
        self.stride = args.stride
        self.save_target = args.save_target
        self.save_cov = args.save_cov
        self.shift_days = args.shift_days
        self.forecast_range = args.forecast_range
        self.operation = args.operation  # or "mean"
        # do we need a separate function for defining the dictionary
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

        # self._check_params()

# Overall procedure
# step 1: parameter Check
# need to do more boundary check for input parameters
# step 2: load data
# (1) get the target variables

    def get_target_data(self, variable_id, target_lat, target_lon, train_date_start, target_end_date, resolution, path):
        """
            Get the target variables with the required lat, lon, date
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
        if resolution == 2:
            # subsample the map with the resolution as 2x2 for target variables
            spatial_map = pd.read_hdf(path + 'target_map_2.h5')
            return spatial_map
        else:
            print('the spatial map for resolution as {} is not available'.format(resolution))

# (3) common functions

    def find_the_cloest_value(self, lat):
        """
            find the cloest lat/lon for our resolution with the given lat/lon
        """
        if lat > math.floor(lat) + 0.5:
            lat_min = math.floor(lat) + 0.75
        else:
            lat_min = math.floor(lat) + 0.25
        return lat_min

    def get_spatial_range(self, lat_range, lon_range):
        """
            Get the subset data for each covariate based on the required spatial and temporal range
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
        # get date range
        # date_index=pd.date_range(start=start_date,end=end_date)
        return lat_index, lon_index

    def remove_masked_data(self, mask_area, lat_range, lon_range, path):
        """
            get the spatial range for us land and ocean mask
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
        """
        Get the correspoding data file from each variable's name
        """
        file_name = path + self.variables_file[variable_id]
        return pd.read_hdf(file_name)

# (2) get the cov variables

    def get_covariates_data_parallel_updated(self, train_date_start, target_end_date, covariate_set, spatial_range, path):
        """
        get the covariate with the required range
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
        """
            Get the subset data for each covariate based on the required temporal range
        """
        date_index = pd.date_range(start=start_date, end=end_date)
        date_index = pd.DataFrame(date_index, columns=['start_date'])
        data_temp = data.reset_index()
        data_subset = date_index.merge(data_temp, on=['start_date'])
        return data_subset

    def get_spatial_subset(self, data, lat_range, lon_range):
        """
            Get the subset data for each covariate based on the required spatial range
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

# (4) create a data frame
    def create_date_data(self, covariates_set, start_date, end_date, path):
        """
            Create a dataframe for all temporal covariates
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
        """
            Create a dataframe for all temporal covariates
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

    def date_adapt(self, train_date_start, test_end_date, past_kyears, past_ndays):
        data_start_date = pd.Timestamp(train_date_start)
        data_start_date = data_start_date - pd.DateOffset(years=past_kyears, days=past_ndays)
        data_end_date = pd.Timestamp(test_end_date)
        # data_end_date = data_end_date + pd.DateOffset(days=shift_days+forecast_range-1)
        return data_start_date, data_end_date

    def shift_target(self, target_df, target_lat, target_lon, target_id, shift_days, forecast_range, operation):
        """
            shift the target variable add compute summation or average values for the required forecast_range
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
        data_start_date = pd.Timestamp(train_date_start)
        data_end_date = pd.Timestamp(test_end_date)
        data_end_date = data_end_date + pd.DateOffset(days=shift_days + forecast_range - 1)
        return data_start_date, data_end_date

# (6) Combination of all operations

    def data_download_target(self):
        # add past years and days to the dataset
        print('compute the real date range considering shift and forecast range')
        train_date_start, target_end_date = self.date_adapt_target(self.train_date_start, self.date_end, self.shift_days, self.forecast_range)
        # get target variable
        print('obtain target data')
        target = self.get_target_data(self.target_variable, self.target_lat, self.target_lon, train_date_start, target_end_date, self.target_res, self.path)
#        print('shift and compute average for target')
        if self.shift_days > 0 or self.forecast_range > 0:
            target = self.shift_target(target, self.target_lat, self.target_lon, self.target_variable, self.shift_days, self.forecast_range, self.operation)

        # get spatial-temporal covariates on land

        if self.save_target is True:
            target.to_hdf('target.h5', key='target', mode='w')
            print('target data saved')
        return target

    def data_download_cov(self):
        # add past years and days to the dataset
        train_date_start, target_end_date = self.date_adapt(self.train_date_start, self.date_end, self.past_kyears, self.past_ndays)
        # get spatial-temporal covariates on land
        print('load data for us continent')
        tic = timeit.default_timer()
        if len(self.covariate_set_us) > 0:
            us_land_spatial_range = self.remove_masked_data('us', self.lat_range_us, self.lon_range_us, self.path)
            covariates_us = self.get_covariates_data_parallel_updated(train_date_start, target_end_date, self.covariate_set_us, us_land_spatial_range, self.path)
            if self.save_cov is True:
                covariates_us.to_hdf('covariates_us.h5', key='covariates_us', mode='w')
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
                covariates_global.to_hdf('covariates_global.h5', key='covariates_global', mode='w')
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
                covariates_sea_pacific, covariates_sea_atlantic = split_pacific_atlantic(self.path, covariates_sea)
                if self.save_cov is True:
                    covariates_sea_pacific.to_hdf('covariates_pacific.h5', key='covariates_pacific', mode='w')
                    covariates_sea_atlantic.to_hdf('covariates_atlantic.h5', key='covariates_atlantic', mode='w')
            else:
                if self.save_cov is True:
                    covariates_sea.to_hdf('covariates_sea.h5', key='covariates_sea', mode='w')
        else:
            covariates_sea = None

        # get spatial covariates
        print('load spatial data')
        if len(self.spatial_set) > 0:
            spatial_covariates = self.create_lat_lon_data(self.spatial_set, self.lat_range_global, self.lon_range_global, self.path)
            # spatial_covariates is not multiindexed, add following
            spatial_covariates.set_index(['lat', 'lon'], inplace=True)
            if self.save_cov is True:
                spatial_covariates.to_hdf('spatial_covariates.h5', key='spatial_covariates', mode='w')
        else:
            spatial_covariates = None

        # get temporal covariates
        print('load temporal data')
        if len(self.temporal_set) > 0:
            temporal_covariates = self.create_date_data(self.temporal_set, train_date_start, self.date_end, self.path)
            if self.save_cov is True:
                temporal_covariates.to_hdf('temporal_covariates.h5', key='temporal_covariates', mode='w')
        else:
            temporal_covariates = None
        return covariates_us, covariates_sea, covariates_global, spatial_covariates, temporal_covariates
