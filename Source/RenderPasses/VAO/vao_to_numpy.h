#pragma once
// supress some warnings for gli
#pragma warning( push )
#pragma warning(disable: 4458) // declaration of 'xxx' hides class member
#include <gli/gli.hpp>
#pragma warning( pop )

#include "npy.h"
#include <iostream>
#include <numeric>

#define FORCE_RAY_OUT_OF_SCREEN 1
#define FORCE_RAY_DOUBLE_SIDED 2
#define FORCE_RAY_INVALID 3

void vao_to_numpy(const std::vector<float> sphereStart,
    std::string raster_image,
    std::string ray_image,
    std::string ask_ray,
    std::string require_ray,
    std::string force_ray,
    std::string raster_ao,
    std::string ray_ao,
    std::string sphere_end,
    int index, bool IsTraining)
{
    static constexpr size_t NUM_SAMPLES = 8;
    
    const bool useDubiousSamples = !IsTraining; // false for training, true for evaluation
    const bool forceDoubleSided = true; // can probably improve the classify accuracy

    gli::texture2d_array texRaster(gli::load(raster_image));
    gli::texture2d_array texRay(gli::load(ray_image));
    //gli::texture2d_array texInstances(gli::load(argv[3]));
    //gli::texture2d_array texCurInstance(gli::load(argv[4]));
    gli::texture2d_array texAskRay(gli::load(ask_ray));
    gli::texture2d_array texRequireRay(gli::load(require_ray));
    gli::texture2d_array texForceRay(gli::load(force_ray));

    gli::texture2d_array texRasterAO(gli::load(raster_ao));
    gli::texture2d_array texRayAO(gli::load(ray_ao));
    gli::texture2d_array texSphereEnd(gli::load(sphere_end));

    auto width = texRaster.extent().x;
    auto height = texRaster.extent().y;
    assert(texRaster.layers() == NUM_SAMPLES);

    // prepare data in float vector
    std::vector<float> rasterSamples;
    std::vector<float> raySamples;
    std::vector<int> pixelXY; // x,y coordinates of pixel
    std::vector<uint8_t> required; // 1 if ray tracing is required, 0 if not (x8)
    std::vector<uint8_t> requiredForced;
    std::vector<uint8_t> asked; // 1 if we want to ask the neural net for a prediction
    std::vector<float> sphereEndSamples;
    std::vector<float> sphereStartSamples;

    //std::vector<int> forcedPixels; // XYi coordinates of the pixel that was forced to be ray traced (or is invalid, in which case the saved ao is 1.0)
    //std::vector<int> numInvalid; // number of invalid samples

    rasterSamples.reserve(width * height * NUM_SAMPLES);
    raySamples.reserve(width * height * NUM_SAMPLES);
    //sameInstance.reserve(width * height * nSamples);
    //rasterInstanceDiffs.reserve(width * height * nSamples);
    pixelXY.reserve(width * height * 2);
    required.reserve(width * height * NUM_SAMPLES);
    asked.reserve(width * height * NUM_SAMPLES);
    sphereEndSamples.reserve(width * height * NUM_SAMPLES);
    sphereStartSamples.reserve(width * height * NUM_SAMPLES);

    //forcedPixels.reserve(width * height * 3 * NUM_SAMPLES);

    // local arrays
    std::array<float, NUM_SAMPLES> raster;
    std::array<float, NUM_SAMPLES> ray;
    std::array<float, NUM_SAMPLES> aoDiff; // difference in AO values
    std::array<uint8_t, NUM_SAMPLES> forceRay;
    std::array<uint8_t, NUM_SAMPLES>  askRay;
    std::array<uint8_t, NUM_SAMPLES> requireRay;
    std::array<float, NUM_SAMPLES> sphereEnd;
    //std::vector<uint32_t> instances;
    //instances.resize(nSamples);

    // fetch function for tex raster (and tex ray)
    auto fetch = gli::detail::convert<gli::texture2d_array, float, gli::defaultp>::call(texRaster.format()).Fetch;
    auto fetchAO = gli::detail::convert<gli::texture2d_array, float, gli::defaultp>::call(texRasterAO.format()).Fetch;
    //auto fetchInt = gli::detail::convert<gli::texture2d_array, float, gli::defaultp>::call(texInstances.format()).Fetch;
    auto fetchBool = gli::detail::convert<gli::texture2d_array, float, gli::defaultp>::call(texForceRay.format()).Fetch;
    int dubiousSamples = 0;
    float equalityThreshold = 0.01f; // assume ray and raster are equal when the values are within this threshold (reduce noise in training data)

    size_t numInvalid = 0; // invalid sample (below the hemisphere)
    size_t numOutOfScreen = 0; // ray tracing was forced
    size_t numDoubleSided = 0;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
    {
        bool outOfScreen = false;
        for (size_t i = 0; i < NUM_SAMPLES; ++i)
        {
            raster[i] = fetch(texRaster, gli::extent2d(x, y), i, 0, 0).r;
            ray[i] = fetch(texRay, gli::extent2d(x, y), i, 0, 0).r;
            sphereEnd[i] = fetch(texSphereEnd, gli::extent2d(x, y), i, 0, 0).r;
            //instances[i] = (uint32_t)fetchInt(texInstances, gli::extent2d(x, y), i, 0, 0).r;
            aoDiff[i] = abs(fetchAO(texRasterAO, gli::extent2d(x, y), i, 0, 0).r - fetchAO(texRayAO, gli::extent2d(x, y), i, 0, 0).r);
            assert(aoDiff[i] <= 1.0f);
            askRay[i] = (uint8_t)fetchBool(texAskRay, gli::extent2d(x, y), i, 0, 0).r;
            requireRay[i] = (uint8_t)fetchBool(texRequireRay, gli::extent2d(x, y), i, 0, 0).r;
            auto forceRayId = int8_t(fetchBool(texForceRay, gli::extent2d(x, y), i, 0, 0).r);
            forceRay[i] = 0;
            if (forceRayId == FORCE_RAY_INVALID)
            {
                numInvalid++;
                forceRay[i] = 1;
            }
            else if (forceRayId == FORCE_RAY_DOUBLE_SIDED && forceDoubleSided)
            {
                numDoubleSided++;
                forceRay[i] = 1;
            }
            else if (forceRayId == FORCE_RAY_OUT_OF_SCREEN)
            {
                numOutOfScreen++;
                outOfScreen = true;
                forceRay[i] = 1;
            }
        }

        if (IsTraining && outOfScreen) continue; // skip out of screen pixels in training data

        // reduce noise
        bool isDubious = false;
        for (size_t i = 0; i < NUM_SAMPLES; ++i)
        {
            // set ray = raster if they are close enough
            if (std::abs(raster[i] - ray[i]) <= equalityThreshold)
            {
                ray[i] = raster[i];
            }
            // clamping for farplane:
            if(raster[i] < -100.0f)
            {
                ray[i] = raster[i]; // clamp ray to raster for very small values (hitting background)
            }

            if (raster[i] < ray[i])
            {
                dubiousSamples++;
                //std::cout << "doubious sample at: " << x << ", " << y << " id: " << i << " diff: " << ray[i] - raster[i] << '\n';
                isDubious = true;
                ray[i] = raster[i]; // set ray to raster
            }
        }

        if (isDubious && !useDubiousSamples) continue; // less noise in training data

        bool noneAsked = std::all_of(askRay.begin(), askRay.end(), [](uint8_t ask) {return ask == 0; });
        if (noneAsked && IsTraining)
            continue; // nothing needs to be evaluated by the neural net

        bool noneForced = std::all_of(forceRay.begin(), forceRay.end(), [](uint8_t force) {return force == 0; });
        if (!IsTraining && noneForced && noneAsked)
            continue; // nothing needs to be evaluated by the neural net
        

        // use this sample
        rasterSamples.insert(rasterSamples.end(), raster.begin(), raster.end());
        raySamples.insert(raySamples.end(), ray.begin(), ray.end());

        pixelXY.push_back(x);
        pixelXY.push_back(y);
        asked.insert(asked.end(), askRay.begin(), askRay.end());
        required.insert(required.end(), requireRay.begin(), requireRay.end());
        requiredForced.insert(requiredForced.end(), forceRay.begin(), forceRay.end());
        sphereEndSamples.insert(sphereEndSamples.end(), sphereEnd.begin(), sphereEnd.end());
        sphereStartSamples.insert(sphereStartSamples.end(), sphereStart.begin(), sphereStart.end());
    }

    const auto strIndex = std::to_string(index);

    // print out number of all samples, empty samples and skipped samples
    std::cout << "Dubious samples (ray > raster): " << dubiousSamples << std::endl;
    unsigned long remainingSamples = (unsigned long)(required.size() / NUM_SAMPLES);

    std::cout << "Remaining samples: " << remainingSamples << std::endl;

    auto numAsked = std::accumulate(asked.begin(), asked.end(), size_t(0));
    auto numRequired = std::accumulate(required.begin(), required.end(), size_t(0));
    std::cout << "Num Asked: " << numAsked << std::endl;
    std::cout << "Num Required: " << numRequired << std::endl;
    std::cout << "Num Double Sided (forced): " << numDoubleSided << std::endl;
    std::cout << "Num Invalid (below hemisphere): " << numInvalid << std::endl;
    std::cout << "Num Excluded because of screen border: " << numOutOfScreen / NUM_SAMPLES << std::endl;


    // write to numpy files
    unsigned long shapeSamples[] = { remainingSamples, (unsigned long)NUM_SAMPLES }; // shape = rows, columns
    unsigned long shapeRequired[] = { remainingSamples, (unsigned long)NUM_SAMPLES };
    unsigned long shapeXY[] = { remainingSamples, 2ul };

    if (IsTraining)
    {
        npy::SaveArrayAsNumpy("raster_train_" + strIndex + ".npy", false, 2, shapeSamples, rasterSamples);
        npy::SaveArrayAsNumpy("ray_train_" + strIndex + ".npy", false, 2, shapeSamples, raySamples);
        npy::SaveArrayAsNumpy("sphere_start_train_" + strIndex + ".npy", false, 2, shapeSamples, sphereStartSamples);
        npy::SaveArrayAsNumpy("sphere_end_train_" + strIndex + ".npy", false, 2, shapeSamples, sphereEndSamples);
        
        npy::SaveArrayAsNumpy("required_train_" + strIndex + ".npy", false, 2, shapeRequired, required);
        npy::SaveArrayAsNumpy("asked_train_" + strIndex + ".npy", false, 2, shapeRequired, asked);
        //npy::SaveArrayAsNumpy("required_forced_train_" + strIndex + ".npy", false, 2, shapeRequired, requiredForced);
    }
    else
    {
        npy::SaveArrayAsNumpy("raster_eval_" + strIndex + ".npy", false, 2, shapeSamples, rasterSamples);
        npy::SaveArrayAsNumpy("ray_eval_" + strIndex + ".npy", false, 2, shapeSamples, raySamples);
        npy::SaveArrayAsNumpy("sphere_start_eval_" + strIndex + ".npy", false, 2, shapeSamples, sphereStartSamples);
        npy::SaveArrayAsNumpy("sphere_end_eval_" + strIndex + ".npy", false, 2, shapeSamples, sphereEndSamples);

        npy::SaveArrayAsNumpy("required_eval_" + strIndex + ".npy", false, 2, shapeRequired, required);
        npy::SaveArrayAsNumpy("asked_eval_" + strIndex + ".npy", false, 2, shapeRequired, asked);
        npy::SaveArrayAsNumpy("required_forced_eval_" + strIndex + ".npy", false, 2, shapeRequired, requiredForced);
        npy::SaveArrayAsNumpy("pixelXY_" + strIndex + ".npy", false, 2, shapeXY, pixelXY);
    }
}

void vao_importance_to_numpy(
    //const std::vector<float> sphereStart,
    std::string raster_image,
    std::string ray_image,
    std::string ask_ray,
    //std::string require_ray,
    std::string force_ray,
    std::string raster_ao,
    std::string ray_ao,
    std::string importance_ao,
    //std::string sphere_end,
    int index, bool IsTraining)
{
    static constexpr size_t NUM_SAMPLES = 8;
    
    const bool includeDoubleSided = true; // can probably improve the classify accuracy

    gli::texture2d_array texRaster(gli::load(raster_image));
    gli::texture2d_array texRay(gli::load(ray_image));

    gli::texture2d_array texAskRay(gli::load(ask_ray));
    gli::texture2d_array texForceRay(gli::load(force_ray));

    gli::texture2d_array texRasterAO(gli::load(raster_ao));
    gli::texture2d_array texRayAO(gli::load(ray_ao));

    gli::texture2d_array texImportance(gli::load(importance_ao));

    auto width = texRasterAO.extent().x;
    auto height = texRasterAO.extent().y;
    assert(texRaster.layers() == NUM_SAMPLES);

    // prepare data in float vector
    std::vector<float> rasterAO;
    std::vector<float> rayAO;
    std::vector<float> radiusImportance;
    std::vector<float> normalImportance;
    std::vector<float> distanceImportance;
    std::vector<float> contributionImportance;

    // fetch function for tex raster (and tex ray)
    auto fetchDepth = gli::detail::convert<gli::texture2d_array, float, gli::defaultp>::call(texRaster.format()).Fetch;
    auto fetchImportance = gli::detail::convert<gli::texture2d_array, float, gli::defaultp>::call(texImportance.format()).Fetch;
    auto fetchAO = gli::detail::convert<gli::texture2d_array, float, gli::defaultp>::call(texRasterAO.format()).Fetch;
    //auto fetchInt = gli::detail::convert<gli::texture2d_array, float, gli::defaultp>::call(texInstances.format()).Fetch;
    auto fetchBool = gli::detail::convert<gli::texture2d_array, float, gli::defaultp>::call(texForceRay.format()).Fetch;
    int dubiousSamples = 0;
    float equalityThreshold = 0.01f; // assume ray and raster are equal when the values are within this threshold (reduce noise in training data)
    
    size_t numOutOfScreen = 0; // ray tracing was forced
    size_t numDoubleSided = 0;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
    {
        bool outOfScreen = false;
        for (size_t i = 0; i < NUM_SAMPLES; ++i)
        {
            uint8_t askRay = (uint8_t)fetchBool(texAskRay, gli::extent2d(x, y), i, 0, 0).r;
            uint8_t forceRay = (uint8_t)fetchBool(texForceRay, gli::extent2d(x, y), i, 0, 0).r;
            if (includeDoubleSided && forceRay == FORCE_RAY_DOUBLE_SIDED)
            {
                askRay = 1;
                numDoubleSided++;
            }
            if (forceRay == FORCE_RAY_OUT_OF_SCREEN)
            {
                numOutOfScreen++;
                continue;
            }
            if (forceRay == FORCE_RAY_INVALID) continue; // skip those

            if (askRay == 0) continue; // this sample was fine (not asked)

            // depths
            //float ray = fetchDepth(texRay, gli::extent2d(x, y), i, 0, 0).r;
            //float raster = fetchDepth(texRaster, gli::extent2d(x, y), i, 0, 0).r;

            float rasterAOvalue = fetchAO(texRasterAO, gli::extent2d(x, y), i, 0, 0).r;
            float rayAOvalue = fetchAO(texRayAO, gli::extent2d(x, y), i, 0, 0).r;

            rasterAO.push_back(rasterAOvalue);
            rayAO.push_back(rayAOvalue);

            glm::vec4 importance = fetchImportance(texImportance, gli::extent2d(x, y), i, 0, 0);
            radiusImportance.push_back(importance.x);
            normalImportance.push_back(importance.y);
            distanceImportance.push_back(importance.z);
            contributionImportance.push_back(importance.w);
        }
    }

    const auto strIndex = std::to_string(index);

    // print out number of all samples, empty samples and skipped samples
    unsigned long remainingSamples = (unsigned long)(rasterAO.size());

    std::cout << "Remaining samples: " << remainingSamples << std::endl;
    std::cout << "Num Double Sided (forced): " << numDoubleSided << std::endl;
    std::cout << "Num Excluded because of screen border: " << numOutOfScreen / NUM_SAMPLES << std::endl;


    // shuffle all arrays with the same seed
    std::shuffle(rasterAO.begin(), rasterAO.end(), std::default_random_engine(0));
    std::shuffle(rayAO.begin(), rayAO.end(), std::default_random_engine(0));
    std::shuffle(radiusImportance.begin(), radiusImportance.end(), std::default_random_engine(0));
    std::shuffle(normalImportance.begin(), normalImportance.end(), std::default_random_engine(0));
    std::shuffle(distanceImportance.begin(), distanceImportance.end(), std::default_random_engine(0));
    std::shuffle(contributionImportance.begin(), contributionImportance.end(), std::default_random_engine(0));

    // write to numpy files
    unsigned long* shape = &remainingSamples;
    
    npy::SaveArrayAsNumpy("rasterao_" + strIndex + ".npy", false, 1, shape, rasterAO);
    npy::SaveArrayAsNumpy("rayao_" + strIndex + ".npy", false, 1, shape, rayAO);

    npy::SaveArrayAsNumpy("radius_importance_" + strIndex + ".npy", false, 1, shape, radiusImportance);
    npy::SaveArrayAsNumpy("normal_importance_" + strIndex + ".npy", false, 1, shape, normalImportance);
    npy::SaveArrayAsNumpy("distance_importance_" + strIndex + ".npy", false, 1, shape, distanceImportance);
    npy::SaveArrayAsNumpy("contribution_importance_" + strIndex + ".npy", false, 1, shape, contributionImportance);
}
