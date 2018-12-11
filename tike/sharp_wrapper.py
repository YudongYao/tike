#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sharp
import numpy as np
import afnumpy as af

class SharpEngine():

    def __init__(self):

        self.eng = sharp.ThrustEngine()  

    def get_sharp_engine(self):

        return self.eng

    def set_data(self, data):

        frames = af.array(np.float32(data[:,:,:]))
        self.eng.setFrames(frames)   

    def set_metadata(self, metadata, data_shape):

        self.eng.m_nframes = data_shape[0]
        self.eng.m_total_n_frames = metadata["all_translations"].shape[0]
        self.eng.m_frame_height = data_shape[1]
        self.eng.m_frame_width = data_shape[2]

        self.eng.setTranslations(af.array(metadata["translations"]))
        self.eng.setAllTranslations(af.array(metadata["all_translations"]))


        if "initial_image" in metadata:
            self.eng.m_image_height = metadata["initial_image"].shape[0]
            self.eng.m_image_width = metadata["initial_image"].shape[1]
            self.eng.m_image = sharp.Range(af.array(metadata["initial_image"], dtype = np.complex64))

        if "detector_mask" in metadata:
            print("Using a detector mask from the input file...")
            self.eng.m_detector_mask = sharp.Range(af.array(metadata["detector_mask"].astype(float), dtype = np.float32))
        else:
            self.eng.setDetectorMask() #this generates a default mask all true
  
        self.eng.setIllumination(sharp.af2thrust(af.array(metadata["illumination"], dtype = np.complex64)))
        self.eng.setReciprocalSize(metadata["reciprocal_res"])

        if "near_field" in  metadata:
            print("Configuring the Engine for a near-field dataset...")
            self.eng.m_near_field = True
            self.eng.m_nf_alpha_radius =  sharp.Range(af.array(metadata["nf_exp_alpha_rad"], dtype = np.complex64))
            #We don't use an illumination mask with near field
            self.eng.m_enforce_illumination_mask = False

        else:
            self.eng.setIlluminationMask(sharp.af2thrust(af.array(metadata["illumination_mask"], dtype = np.complex64)))

        if "image_regularization_term" in  metadata:
            self.eng.m_image_regularized =  sharp.Range(af.array(metadata[image_regularization_term], dtype = np.complex64)) 

        if "image_regularization_factor" in  metadata:
            self.eng.setRegularizerImage(metadata["image_regularization_factor"])


    def set_options(self, options):

        print(self.eng.m_solver)

        if "illumination_ref" in options:
            print("Illumination refinement every " + str(options["illumination_ref"]) + " iterations.")
            self.eng.setIlluminationRefinement(options["illumination_ref"])

        if "background_ref" in options:
            print("Background refinement every " + str(options["background_ref"]) + " iterations.")
            self.eng.setBackgroundRefinement(options["background_ref"])

        if "bck_type" in options:
            self.eng.setBackgroundType(options["bck_type"])

        if "solver" in options:
            print("Using " + str(options["solver"]) + " solver.")

            if options["solver"] is "RAAR":
                self.eng.setRAAR()
                self.eng.m_beta = 0.6                
                if "RAAR_beta" in options:
                    print("RAAR beta used = " + str(options["RAAR_beta"]))
                    self.eng.m_beta = options["RAAR_beta"]

            if options["solver"] is "ADMM":
                self.eng.setADMM()
                if "ADMM_r" in options:
                    print("ADMM r factor used = " + str(options["ADMM_r"])) 
                    self.eng.m_ADMM_penalty_1 = options["ADMM_r"]   

    def initialize(self):

        self.eng.initialize()

    def reconstruct(self, iterations):

        self.eng.iterate(iterations)

        image = sharp.af_sharp.range2af(self.eng.getImage())
        illumination = sharp.af_sharp.range2af(self.eng.getIllumination())
        corners = sharp.af_sharp.range2af(self.eng.m_frames_corners)

        image_cpu = np.array(image)
        illumination_cpu = np.array(illumination).reshape(self.eng.m_frame_height, self.eng.m_frame_width) #sharp gives illumination in 1D
        
        return image_cpu, illumination_cpu


