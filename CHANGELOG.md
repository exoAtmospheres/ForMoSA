# Changelog

Auto-generated from commit history, grouped by author. Not hand-maintained -- see `git log` for full detail.

## [Unreleased]

### Allan Denis
- small correction ([680bd6d](https://github.com/exoAtmospheres/ForMoSA/commit/680bd6df2d6cca3ad010bc439430ef943543872f))
- Fixing issue in unnormalized ccf which was returned by the method compute_ccf of ns_analysis.py ([483360f](https://github.com/exoAtmospheres/ForMoSA/commit/483360f3aff48c1ead09dabf11a6151b0780247c))
- Fixing issue with loading results from a previous ForMoSA run ([e2745a0](https://github.com/exoAtmospheres/ForMoSA/commit/e2745a0de3f5bd3fa99594ec716d5c6fbc574a02))
- same modifications as previously but for filter ([4998b9e](https://github.com/exoAtmospheres/ForMoSA/commit/4998b9ebab8f4e1378dc7e33d44dccb5e328f096))
- correctly propagating insturment and facility to the length of wave when user gives only one value for these parameters + factorizing ([1ced228](https://github.com/exoAtmospheres/ForMoSA/commit/1ced228b560b69b266e6e9d3bf824febc72f0ac2))
- load results from json directly in the NSResults class ([5df0711](https://github.com/exoAtmospheres/ForMoSA/commit/5df0711211c4f49c98b6834399a3d0ae15f52f17))
- adding loglikelihood type to ccf computation ([d1bbd96](https://github.com/exoAtmospheres/ForMoSA/commit/d1bbd9631e813abc44d3316a7fdcfe06e0f45a31))
- niormalize properly observations and models in logL_CCF_Brogi ([94cb72c](https://github.com/exoAtmospheres/ForMoSA/commit/94cb72c91c0b140177166762992dcef7ad99cc93))
- some updates on ObservedModel ([bdc41ea](https://github.com/exoAtmospheres/ForMoSA/commit/bdc41eab0e5c06c9c3f50b6b7418468c54c55ebe))
- factorizing loglilkelihood computation ([e7f77a3](https://github.com/exoAtmospheres/ForMoSA/commit/e7f77a30fcb1d4dba25b445427466717429d7ffe))
- Adding loglikelihood type in ccf computation ([1cd1c8e](https://github.com/exoAtmospheres/ForMoSA/commit/1cd1c8ecabe67713eddea884289a52917312cb9b))
- fixing issue when interpolating grid filled with nan values ([1ae9afb](https://github.com/exoAtmospheres/ForMoSA/commit/1ae9afb0310f92ca58e2881f6731fc3d0caf8306))
- small correction + fixing doppler shift which was inverted ([4f674f0](https://github.com/exoAtmospheres/ForMoSA/commit/4f674f0698417858ff0d49614c0d2eae7e88c962))
- Adding a safety normalization of labels in property labels of observation class ([db59aca](https://github.com/exoAtmospheres/ForMoSA/commit/db59aca753fb3b36c7d798dc0da34e5cea836fb5))
- Using figsize of the config for plot_fit ([0a94ad4](https://github.com/exoAtmospheres/ForMoSA/commit/0a94ad48e78c2e125e100d110398594805d6682d))
- retrieving the true figure size given by the config in radar_plot ([d1cbb0f](https://github.com/exoAtmospheres/ForMoSA/commit/d1cbb0f0db4be6c675c4058636e9914e939753b2))
- Implementation of name and labels attributes in observation class ([074af4a](https://github.com/exoAtmospheres/ForMoSA/commit/074af4aad2daeb476e052a35b2624d545c2e36f1))
- Taking fixed parameters into account ([443b0da](https://github.com/exoAtmospheres/ForMoSA/commit/443b0da8534a7bc776fdc31a57eef0e3f174d35e))
- Fixing issue ([dcbe5f0](https://github.com/exoAtmospheres/ForMoSA/commit/dcbe5f0cf684d462329efb8222e4763018516712))
- Revert "fixing issue in multi-observations parameters" ([1b70aa5](https://github.com/exoAtmospheres/ForMoSA/commit/1b70aa5f00c4418e95cc58d1dfe6e06575360ad7))
- fixing issue in multi-observations parameters ([d1a86e6](https://github.com/exoAtmospheres/ForMoSA/commit/d1a86e62219e4e12dc72011b3d65dfbb50c8f04a))
- Fixing issue in saving observation as fits file ([c69fed4](https://github.com/exoAtmospheres/ForMoSA/commit/c69fed4775a9c04cfbf7e37c932052323c42cc28))
- adding filter integration on utils ([976f0be](https://github.com/exoAtmospheres/ForMoSA/commit/976f0beae64078865330760a98e8570ded50173a))
- Revert "Fixing issue in soving observation as fits.file" ([e337140](https://github.com/exoAtmospheres/ForMoSA/commit/e33714012d0ffa964a92ca580ab18a28a39de372))
- fixing issue in multi-observation parameters ([1a74796](https://github.com/exoAtmospheres/ForMoSA/commit/1a747967141ae3c60213b55ec053ad491676fa1d))
- Fixing issue in soving observation as fits.file ([ae89d48](https://github.com/exoAtmospheres/ForMoSA/commit/ae89d48f5eb34a8266ae827ae5d3c3435ef9b232))

### Bhavesh Rajpoot
- BR: Revise README for environment setup and version changes ([c9deaa4](https://github.com/exoAtmospheres/ForMoSA/commit/c9deaa47fd85a90950d3db91c346a731bcc839f5))
- BR: Bump version from 2.0.2 to 2.1.0 ([62bf9ed](https://github.com/exoAtmospheres/ForMoSA/commit/62bf9ed47a057aa5682664b8e528b46d64b4346f))
- BR: Fix the obs color in best fit plot ([1ea8929](https://github.com/exoAtmospheres/ForMoSA/commit/1ea8929f10a80fe3fd6fd2567855cbce7dfb8f8f))
- BR: Completed the advanced plotting tutorial ([33423c6](https://github.com/exoAtmospheres/ForMoSA/commit/33423c60092d92073e5f6ec0e21bc192a3fe2e87))
- BR: Add missing av field to ConfigParameters (fixes #34) ([fcb78bc](https://github.com/exoAtmospheres/ForMoSA/commit/fcb78bc020fef6c8c2e0f3216258c27ef418e000))
- BR: Reconcile docs with the changelog automation and current v2.0 API ([8b57455](https://github.com/exoAtmospheres/ForMoSA/commit/8b574551808e7b6742007d856b4d25511038a84d))
- BR: Added commit-based changelog ([3f24840](https://github.com/exoAtmospheres/ForMoSA/commit/3f24840ea58dcce3feb60581f68896f73e12d741))
- BR: Fix Keck/NIRC2.Lp filter case typo, remove test's SVO network dependency ([f58eff2](https://github.com/exoAtmospheres/ForMoSA/commit/f58eff29bb115b58cb13d8781e557f880b23de16))
- BR: Relax doc-tooling version floors for CI compatibility ([02d946d](https://github.com/exoAtmospheres/ForMoSA/commit/02d946deb5436c727466276be324f24fd425b463))
- BR: Ensure observation names are unique in ObservationSet (fixes #23) ([cbcb4b8](https://github.com/exoAtmospheres/ForMoSA/commit/cbcb4b8857db881facaba1a5b8aca55967f1bddc))
- BR: Fix calculation of res_mod_obs_spectro_broad ([e163c22](https://github.com/exoAtmospheres/ForMoSA/commit/e163c227790e623fd3379893307fef45f417007a))
- BR: Update weights calculation in fit_linear_model ([2eb6921](https://github.com/exoAtmospheres/ForMoSA/commit/2eb69212678f687109462316a88a992abf82fa67))
- BR: Add physics-effect unit tests, fix bugs found along the way ([f46b3a6](https://github.com/exoAtmospheres/ForMoSA/commit/f46b3a698875e90b68f7916d02f1f62a535554e5))
- BR: fixing the sphinx version ([23edda9](https://github.com/exoAtmospheres/ForMoSA/commit/23edda97b1ce7a5493f49fb39a184245d7fbf1f3))
- BR: Fix formatting in pyproject.toml for authors list ([39447d0](https://github.com/exoAtmospheres/ForMoSA/commit/39447d0cc9b3bb701ce2e032707331fb282ff0f9))
- BR: Updating gitignore ([6915311](https://github.com/exoAtmospheres/ForMoSA/commit/69153115278d596b15250c7349450c456cd2f474))
- BR: Branch update in github tests action ([af5606b](https://github.com/exoAtmospheres/ForMoSA/commit/af5606bc537857beb85b45e9b8bfd8a04003526b))
- BR: MOSAIC Tutorial ([3bd0682](https://github.com/exoAtmospheres/ForMoSA/commit/3bd068247be8527d441e1111cd94d9c55e74be76))
- BR: Update build backend for setuptools in pyproject.toml ([86eabe4](https://github.com/exoAtmospheres/ForMoSA/commit/86eabe45abcc1baf46327ad0c2074e994aa1f36f))
- BR: removing agentic files ([7e712b7](https://github.com/exoAtmospheres/ForMoSA/commit/7e712b76120998fbbfcd71236b449eafc34ec4d7))
- BR: updated data download paths in demos ([2aa97cd](https://github.com/exoAtmospheres/ForMoSA/commit/2aa97cd24d7c987dfafd960e0ac9ba6689c904ef))

### Paulina Palma-Bifani
- Update Matthieu Ravet affiliation in paper and recompile PDF ([555dc0e](https://github.com/exoAtmospheres/ForMoSA/commit/555dc0e777dcc5764a409467da32286002d09e6e))
- Update paper author affiliations and references ([477aa6c](https://github.com/exoAtmospheres/ForMoSA/commit/477aa6c8df2cb14450427d4d7fcdf5f25183cc94))
- Update Gaël Chauvin affiliation, switch to paper.bib, and recompile PDF ([ba74385](https://github.com/exoAtmospheres/ForMoSA/commit/ba74385cbb798d1c8a7d05ec64e9b9830bf42583))
- Update equal contribution flags for authors and recompile PDF ([41ec517](https://github.com/exoAtmospheres/ForMoSA/commit/41ec517d5185d066ebed17fbdd97a6ce2980dd28))
- Update paper text structure and recompile PDF ([1af313f](https://github.com/exoAtmospheres/ForMoSA/commit/1af313f71c481911dfe43765222d0ad89819704a))
- Update ForMoSA Collaboration affiliations and compile final PDF draft ([945ee10](https://github.com/exoAtmospheres/ForMoSA/commit/945ee1025448b566ea7adf8a9f8967fccc38e689))
- Fix overfull box warning and add flat author list for local PDF preview ([c21897d](https://github.com/exoAtmospheres/ForMoSA/commit/c21897d20f3763acd700ee5369ba63d28382f182))
- Update paper draft, clean up bibliography database, and fix compilation warnings ([980a984](https://github.com/exoAtmospheres/ForMoSA/commit/980a98466f680f4625debe4a6067881862df086f))

### Bhavesh012
- chore: update changelog [skip ci] ([16ebda6](https://github.com/exoAtmospheres/ForMoSA/commit/16ebda6fd138b75bacba44679b9cca8b8a6262fc))
- chore: update changelog [skip ci] ([2097326](https://github.com/exoAtmospheres/ForMoSA/commit/2097326809d8a69fab67e28a1905a83b726e0d6a))
- chore: update changelog [skip ci] ([cd7add6](https://github.com/exoAtmospheres/ForMoSA/commit/cd7add6324b616f48fb2ac094eebbaa089f470bf))
- chore: update changelog [skip ci] ([b4a4f0d](https://github.com/exoAtmospheres/ForMoSA/commit/b4a4f0da5aeb3ba8a7c2b0b3ec42debb8d2aa386))

### Matthieu Ravet
- MR: Updating docs with paper sections Performances & Accuracy ([73d98dc](https://github.com/exoAtmospheres/ForMoSA/commit/73d98dc83870233d39698503222eb1ee60f8c9c9))

