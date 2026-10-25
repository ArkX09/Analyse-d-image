#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ----------------------------------------------------------------------------
// Detecteurs
// ----------------------------------------------------------------------------
static cv::Mat lisser(const cv::Mat& gray, double sigma)
{
    cv::Mat g;
    if (sigma > 0) cv::GaussianBlur(gray, g, cv::Size(), sigma);
    else g = gray.clone();
    return g;
}

static cv::Mat detectSobel(const cv::Mat& gray, int seuil, int ksize, double sigma)
{
    cv::Mat g = lisser(gray, sigma), gx, gy, mag;
    cv::Sobel(g, gx, CV_32F, 1, 0, ksize);
    cv::Sobel(g, gy, CV_32F, 0, 1, ksize);
    cv::magnitude(gx, gy, mag);
    double mx;
    cv::minMaxLoc(mag, nullptr, &mx);
    if (mx > 0) mag = mag * (255.0 / mx);
    cv::Mat out;
    cv::threshold(mag, out, seuil, 1, cv::THRESH_BINARY);
    out.convertTo(out, CV_8U);
    return out;
}

static cv::Mat detectLaplace(const cv::Mat& gray, int seuil, int ksize, double sigma)
{
    cv::Mat g = lisser(gray, sigma), lap;
    cv::Laplacian(g, lap, CV_32F, ksize);
    lap = cv::abs(lap);
    double mx;
    cv::minMaxLoc(lap, nullptr, &mx);
    if (mx > 0) lap = lap * (255.0 / mx);
    cv::Mat out;
    cv::threshold(lap, out, seuil, 1, cv::THRESH_BINARY);
    out.convertTo(out, CV_8U);
    return out;
}

static cv::Mat detectCanny(const cv::Mat& gray, int bas, int haut, double sigma)
{
    cv::Mat g = lisser(gray, sigma), e;
    cv::Canny(g, e, bas, haut);
    cv::Mat out;
    cv::threshold(e, out, 0, 1, cv::THRESH_BINARY);
    return out;
}

// ----------------------------------------------------------------------------
// Metriques
// ----------------------------------------------------------------------------
struct Compte {
    long detectes = 0, reference = 0, corrects = 0, fp = 0, fn = 0;
};

struct Mesures {
    double P = 0, TFP = 0, TFN = 0;
};

static Compte comparer(const cv::Mat& detecte, const cv::Mat& reference, int tol)
{
    Compte c;
    c.detectes = cv::countNonZero(detecte);
    c.reference = cv::countNonZero(reference);
    if (tol <= 0) {
        cv::Mat inter;
        cv::bitwise_and(detecte, reference, inter);
        c.corrects = cv::countNonZero(inter);
        c.fp = c.detectes - c.corrects;
        c.fn = c.reference - c.corrects;
    }
    else {
        cv::Mat k = cv::Mat::ones(2 * tol + 1, 2 * tol + 1, CV_8U);
        cv::Mat refD, detD, detOK, refOK;
        cv::dilate(reference, refD, k);
        cv::dilate(detecte, detD, k);
        cv::bitwise_and(detecte, refD, detOK);     // detectes proches d'une reference
        c.fp = c.detectes - cv::countNonZero(detOK);
        cv::bitwise_and(reference, detD, refOK);   // references retrouvees par une detection proche
        c.corrects = cv::countNonZero(refOK);
        c.fn = c.reference - c.corrects;
    }
    return c;
}

static Mesures mesures(const Compte& c)
{
    Mesures m;
    double tot = double(c.corrects + c.fp + c.fn);
    if (tot > 0) {
        m.P = c.corrects / tot;
        m.TFP = c.fp / tot;
        m.TFN = c.fn / tot;
    }
    return m;
}

// ----------------------------------------------------------------------------
// Fichiers
// ----------------------------------------------------------------------------
static std::string minuscules(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

static std::vector<fs::path> listerImages(const std::string& dossier)
{
    static const std::vector<std::string> exts = { ".png", ".jpg", ".jpeg", ".bmp", ".pgm",
                                                  ".ppm"};
    std::vector<fs::path> res;
    if (!fs::is_directory(dossier)) return res;
    for (const auto& e : fs::directory_iterator(dossier)) {
        if (!e.is_regular_file()) continue;
        std::string ext = minuscules(e.path().extension().string());
        if (std::find(exts.begin(), exts.end(), ext) != exts.end()) res.push_back(e.path());
    }
    std::sort(res.begin(), res.end());
    return res;
}

static bool trouverReference(const fs::path& image, const std::string& dossierRefs, fs::path& ref)
{
    std::string base = image.stem().string();
    for (const auto& f : listerImages(dossierRefs)) {
        std::string b = f.stem().string();
        if (b == base || b.rfind(base, 0) == 0 || base.rfind(b, 0) == 0) {
            ref = f;
            return true;
        }
    }
    return false;
}

// Charge une image de contours de reference -> binaire 0/1, contours = pixels minoritaires
static cv::Mat chargerReference(const fs::path& chemin, cv::Size taille)
{
    cv::Mat ref = cv::imread(chemin.string(), cv::IMREAD_GRAYSCALE);
    if (ref.empty()) return ref;
    if (ref.size() != taille) cv::resize(ref, ref, taille, 0, 0, cv::INTER_NEAREST);
    cv::Mat b;
    cv::threshold(ref, b, 127, 1, cv::THRESH_BINARY);
    if (cv::mean(b)[0] > 0.5) b = 1 - b;   // contours noirs sur fond blanc -> on inverse
    return b;
}

// ----------------------------------------------------------------------------
// Arguments
// ----------------------------------------------------------------------------
struct Options {
    std::string mode, images, refs, out = "resultats";
    int tol = 0;
    double sigma = 1.4;
};

static bool lireArguments(int argc, char** argv, Options& o)
{
    if (argc < 2) return false;
    o.mode = argv[1];
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--images") o.images = v;
        else if (k == "--refs") o.refs = v;
        else if (k == "--out") o.out = v;
        else if (k == "--tol") o.tol = std::stoi(v);
        else if (k == "--sigma") o.sigma = std::stod(v);
        else { std::cerr << "Option inconnue : " << k << "\n"; return false; }
    }
    return (o.mode == "interactif" || o.mode == "evaluer") && !o.images.empty();
}

// ----------------------------------------------------------------------------
// Mode interactif : trackbars
// ----------------------------------------------------------------------------
static int modeInteractif(const Options& o)
{
    auto fichiers = listerImages(o.images);
    if (fichiers.empty()) {
        std::cerr << "Aucune image trouvee dans " << o.images << "\n";
        return 1;
    }

    for (const auto& f : fichiers) {
        cv::Mat gray = cv::imread(f.string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty()) continue;

        cv::Mat ref;
        if (!o.refs.empty()) {
            fs::path rf;
            if (trouverReference(f, o.refs, rf)) ref = chargerReference(rf, gray.size());
        }

        // Valeurs controlees par les trackbars
        int sSeuil = 60, sK = 1, sSig = 14;          // Sobel
        int lSeuil = 20, lK = 1, lSig = 14;          // Laplace
        int cBas = 50, cHaut = 150, cSig = 14;       // Canny

        cv::namedWindow("Original");
        cv::namedWindow("Sobel");
        cv::namedWindow("Laplace");
        cv::namedWindow("Canny");
        cv::createTrackbar("seuil", "Sobel", &sSeuil, 255);
        cv::createTrackbar("ksize 1,3,5,7", "Sobel", &sK, 3);
        cv::createTrackbar("sigma x10", "Sobel", &sSig, 50);
        cv::createTrackbar("seuil", "Laplace", &lSeuil, 255);
        cv::createTrackbar("ksize 1,3,5,7", "Laplace", &lK, 3);
        cv::createTrackbar("sigma x10", "Laplace", &lSig, 50);
        cv::createTrackbar("seuil bas", "Canny", &cBas, 255);
        cv::createTrackbar("seuil haut", "Canny", &cHaut, 255);
        cv::createTrackbar("sigma x10", "Canny", &cSig, 50);
        if (!ref.empty()) {
            cv::namedWindow("Reference");
            cv::imshow("Reference", ref * 255);
        }

        std::cout << "\n[" << f.filename().string()
            << "]  q = image suivante, ESC = quitter, p = afficher P/TFP/TFN\n";
        while (true) {
            cv::Mat s = detectSobel(gray, sSeuil, 2 * sK + 1, sSig / 10.0);
            cv::Mat l = detectLaplace(gray, lSeuil, 2 * lK + 1, lSig / 10.0);
            int haut = std::max(cHaut, cBas + 1);
            cv::Mat c = detectCanny(gray, cBas, haut, cSig / 10.0);

            cv::imshow("Original", gray);
            cv::imshow("Sobel", s * 255);
            cv::imshow("Laplace", l * 255);
            cv::imshow("Canny", c * 255);

            int k = cv::waitKey(30) & 0xFF;
            if (k == 'p' && !ref.empty()) {
                const std::pair<const char*, const cv::Mat*> liste[] = {
                    { "Sobel", &s },{ "Laplace", &l },{ "Canny", &c } };
                for (const auto& d : liste) {
                    Mesures m = mesures(comparer(*d.second, ref, o.tol));
                    std::cout << "  " << std::left << std::setw(8) << d.first << std::fixed
                        << std::setprecision(3) << " P=" << m.P << "  TFP=" << m.TFP
                        << "  TFN=" << m.TFN << "\n";
                }
            }
            if (k == 'q') break;
            if (k == 27) { cv::destroyAllWindows(); return 0; }
        }
        cv::destroyAllWindows();
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Mode evaluation
// ----------------------------------------------------------------------------
struct Param {
    int a = 0, b = 0;     // Sobel/Laplace : a = seuil, b = ksize ; Canny : a = bas, b = haut
    std::string texte;
};

static std::vector<Param> grille(const std::string& det)
{
    std::vector<Param> g;
    if (det == "Sobel") {
        for (int k : {3, 5})
            for (int t = 10; t < 200; t += 10)
                g.push_back({ t, k, "seuil=" + std::to_string(t) + " ksize=" + std::to_string(k) });
    }
    else if (det == "Laplace") {
        for (int k : {3, 5})
            for (int t = 2; t < 60; t += 3)
                g.push_back({ t, k, "seuil=" + std::to_string(t) + " ksize=" + std::to_string(k) });
    }
    else {
        for (int bas = 10; bas <= 200; bas += 10)
            for (int ratio : {2, 3})
                g.push_back({ bas, bas * ratio,
                             "bas=" + std::to_string(bas) + " haut=" + std::to_string(bas * ratio) });
    }
    return g;
}

static cv::Mat appliquer(const std::string& det, const cv::Mat& gray, const Param& p, double sigma)
{
    if (det == "Sobel") return detectSobel(gray, p.a, p.b, sigma);
    if (det == "Laplace") return detectLaplace(gray, p.a, p.b, sigma);
    return detectCanny(gray, p.a, p.b, sigma);
}

struct Donnee {
    std::string nom;
    cv::Mat gray, ref;
};

static int modeEvaluer(const Options& o)
{
    if (o.refs.empty()) {
        std::cerr << "--refs est obligatoire pour le mode evaluer\n";
        return 1;
    }
    fs::create_directories(o.out);

    std::vector<Donnee> data;
    for (const auto& f : listerImages(o.images)) {
        fs::path rf;
        if (!trouverReference(f, o.refs, rf)) continue;
        Donnee d;
        d.nom = f.filename().string();
        d.gray = cv::imread(f.string(), cv::IMREAD_GRAYSCALE);
        if (d.gray.empty()) continue;
        d.ref = chargerReference(rf, d.gray.size());
        if (d.ref.empty()) continue;
        data.push_back(d);
    }
    if (data.empty()) {
        std::cerr << "Aucune paire image / reference trouvee (verifie les noms de fichiers).\n";
        return 1;
    }
    std::cout << data.size() << " paires image/reference\n";

    std::ofstream fImg(o.out + "/resultats_par_image.csv");
    fImg << "detecteur;image;parametres;contours_detectes;contours_reference;contours_corrects;"
        "faux_positifs;faux_negatifs;P;TFP;TFN\n";
    std::ofstream fRes(o.out + "/resume.csv");
    fRes << "detecteur;strategie;parametres;P_moyen;TFP_moyen;TFN_moyen\n";
    fImg << std::fixed << std::setprecision(4);
    fRes << std::fixed << std::setprecision(4);

    std::cout << "\n" << std::left << std::setw(9) << "Detect." << std::setw(22) << "Strategie"
        << std::right << std::setw(8) << "P" << std::setw(8) << "TFP" << std::setw(8) << "TFN"
        << "   Parametres\n";

    for (const std::string det : {"Sobel", "Laplace", "Canny"}) {
        std::vector<Param> params = grille(det);
        const size_t nP = params.size(), nI = data.size();

        // res[i][j] : parametre i, image j
        std::vector<std::vector<Compte>> res(nP, std::vector<Compte>(nI));
        for (size_t i = 0; i < nP; ++i)
            for (size_t j = 0; j < nI; ++j)
                res[i][j] = comparer(appliquer(det, data[j].gray, params[i], o.sigma), data[j].ref, o.tol);

        // --- Parametres globaux : meilleur P moyen sur toute la base
        size_t ib = 0;
        double bestP = -1;
        for (size_t i = 0; i < nP; ++i) {
            double s = 0;
            for (size_t j = 0; j < nI; ++j) s += mesures(res[i][j]).P;
            s /= nI;
            if (s > bestP) { bestP = s; ib = i; }
        }
        Mesures mg;
        for (size_t j = 0; j < nI; ++j) {
            Mesures m = mesures(res[ib][j]);
            mg.P += m.P / nI; mg.TFP += m.TFP / nI; mg.TFN += m.TFN / nI;
        }

        // --- Meilleur parametre image par image (borne superieure)
        Mesures mb;
        for (size_t j = 0; j < nI; ++j) {
            size_t ij = 0;
            double pj = -1;
            for (size_t i = 0; i < nP; ++i) {
                double p = mesures(res[i][j]).P;
                if (p > pj) { pj = p; ij = i; }
            }
            const Compte& c = res[ij][j];
            Mesures m = mesures(c);
            mb.P += m.P / nI; mb.TFP += m.TFP / nI; mb.TFN += m.TFN / nI;
            fImg << det << ';' << data[j].nom << ';' << params[ij].texte << ';' << c.detectes << ';'
                << c.reference << ';' << c.corrects << ';' << c.fp << ';' << c.fn << ';' << m.P
                << ';' << m.TFP << ';' << m.TFN << '\n';
            cv::Mat out = appliquer(det, data[j].gray, params[ij], o.sigma) * 255;
            std::string base = fs::path(data[j].nom).stem().string();
            cv::imwrite(o.out + "/" + base + "_" + det + ".png", out);
        }

        fRes << det << ";parametres globaux;" << params[ib].texte << ';' << mg.P << ';' << mg.TFP
            << ';' << mg.TFN << '\n';
        fRes << det << ";meilleur par image;-;" << mb.P << ';' << mb.TFP << ';' << mb.TFN << '\n';

        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::left << std::setw(9) << det << std::setw(22) << "parametres globaux"
            << std::right << std::setw(8) << mg.P << std::setw(8) << mg.TFP << std::setw(8)
            << mg.TFN << "   " << params[ib].texte << "\n";
        std::cout << std::left << std::setw(9) << det << std::setw(22) << "meilleur par image"
            << std::right << std::setw(8) << mb.P << std::setw(8) << mb.TFP << std::setw(8)
            << mb.TFN << "\n";
    }
    std::cout << "\nFichiers ecrits dans : " << o.out << "/\n";
    return 0;
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    Options o;
    if (!lireArguments(argc, argv, o)) {
        std::cerr << "Usage :\n"
            "  " << argv[0] << " interactif --images dossier [--refs dossier] [--tol N]\n"
            "  " << argv[0] << " evaluer --images dossier --refs dossier [--out dossier] [--tol N] [--sigma S]\n";
        return 1;
    }
    return o.mode == "interactif" ? modeInteractif(o) : modeEvaluer(o);
}