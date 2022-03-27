#include <cmath>

namespace lumina {

    struct dim {

        float x, y, z;

        dim(float x = 0.f, float y = 0.f, float z = 0.f)
            : x(x), y(y), z(z) {
        }

        dim(const dim&) = default;

        dim(dim&&) = default;

        dim& operator = (const dim&) = default;

        dim& operator = (dim&&) = default;

        dim& operator += (const dim& v) {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        dim& operator -= (const dim& v) {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }

        dim& operator *= (const float f) {
            x *= f;
            y *= f;
            z *= f;
            return *this;
        }

        dim& operator /= (const float f) {
            x /= f;
            y /= f;
            z /= f;
            return *this;
        }

        dim operator + (const dim& v) const {
            return dim(*this) += v;
        }

        dim operator - (const dim& v) const {
            return dim(*this) -= v;
        }

        dim operator * (const float f) const {
            return dim(*this) *= f;
        }

        dim operator / (const float f) const {
            return dim(*this) *= f;
        }

        float len() const {
            return std::sqrt(x * x + y * y + z * z);
        }

        dim normal() const {
            float l = this->len();
            return dim(x / l, y / l, z / l);
        }

        static float dot(const dim& v1, const dim& v2) {
            return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        }

        static dim cross(const dim& v1, const dim& v2) {
            return dim();
        }

    };

    struct triangle {

        dim v1, v2, v3;

        triangle(const dim& v1 = dim(), const dim& v2 = dim(), const dim& v3 = dim())
            : v1(v1), v2(v2), v3(v3) {
        }

        dim normal() {
            return dim::cross(v2 - v1, v3 - v1);
        }

    };

}