/*
 * Copyright (c) 2016-present Samsung Electronics Co., Ltd
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
 *  USA
 */

#ifndef __EscargotBasicString__
#define __EscargotBasicString__

#include "Escargot.h"
#include <cstddef>
#include <cstring>
#include <algorithm>

namespace Escargot {

template <typename T, typename Allocator>
class BasicString : public gc {
    static T* emptyBuffer()
    {
        static T e[1] = { 0 };
        return e;
    }

    void makeEmpty()
    {
        m_buffer = emptyBuffer();
        m_size = 0;
        m_capacity = 0;
    }

    void ensureCapacity(size_t required)
    {
        if (required <= m_capacity)
            return;

        size_t newCapacity = m_capacity ? std::max(required, m_capacity + (m_capacity >> 1) + 1) : required;
        T* newBuffer = allocate(newCapacity);
        if (m_size) {
            memcpy(newBuffer, m_buffer, sizeof(T) * m_size);
        }
        if (m_buffer != emptyBuffer())
            deallocate(m_buffer, m_capacity);
        m_buffer = newBuffer;
        m_capacity = newCapacity;
        m_buffer[m_size] = 0;
    }

public:
    BasicString()
    {
        makeEmpty();
    }

    BasicString(const T* src, size_t len)
        : m_buffer(allocate(len))
        , m_size(len)
        , m_capacity(len)
    {
        memcpy(m_buffer, src, sizeof(T) * m_size);
        m_buffer[m_size] = 0;
    }

    BasicString(BasicString<T, Allocator>&& other)
        : m_buffer(other.m_buffer)
        , m_size(other.size())
        , m_capacity(other.m_capacity)
    {
        other.makeEmpty();
    }

    BasicString(const BasicString<T, Allocator>& other)
    {
        if (other.size()) {
            m_size = other.size();
            m_capacity = other.m_capacity ? other.m_capacity : other.m_size;
            m_buffer = allocate(m_capacity);
            memcpy(m_buffer, other.data(), sizeof(T) * m_size);
            m_buffer[m_size] = 0;
        } else {
            makeEmpty();
        }
    }

    const BasicString<T, Allocator>& operator=(const BasicString<T, Allocator>& other)
    {
        if (&other == this)
            return *this;

        if (other.size()) {
            size_t newSize = other.size();
            ensureCapacity(other.m_capacity ? other.m_capacity : newSize);
            m_size = newSize;
            memcpy(m_buffer, other.data(), sizeof(T) * m_size);
            m_buffer[m_size] = 0;
        } else {
            clear();
        }
        return *this;
    }

    ~BasicString()
    {
        if (m_buffer != emptyBuffer())
            deallocate(m_buffer, m_capacity);
    }

    void pushBack(const T& val)
    {
        ensureCapacity(m_size + 1);
        m_buffer[m_size] = val;
        m_size++;
        m_buffer[m_size] = 0;
    }

    void push_back(const T& val)
    {
        pushBack(val);
    }

    void append(const T* src, size_t len)
    {
        size_t newLen = m_size + len;
        ensureCapacity(newLen);
        memcpy(m_buffer + m_size, src, len * sizeof(T));
        m_size = newLen;
        m_buffer[m_size] = 0;
    }

    void insert(size_t pos, const T& val)
    {
        ASSERT(pos < m_size);
        ensureCapacity(m_size + 1);
        for (size_t i = m_size; i > pos; i--) {
            m_buffer[i] = m_buffer[i - 1];
        }
        m_buffer[pos] = val;
        m_size++;
        m_buffer[m_size] = 0;
    }

    void erase(size_t pos)
    {
        erase(pos, pos + 1);
    }

    void erase(size_t start, size_t end)
    {
        ASSERT(start < end);
        ASSERT(end <= m_size);

        size_t c = end - start;
        size_t newSize = m_size - c;
        if (newSize) {
            if (m_size > end) {
                for (size_t i = end; i < m_size; i++) {
                    m_buffer[i - c] = m_buffer[i];
                }
            }
            m_size = newSize;
            m_buffer[m_size] = 0;
        } else {
            clear();
        }
    }

    size_t size() const
    {
        return m_size;
    }

    size_t length() const
    {
        return m_size;
    }

    size_t capacity() const
    {
        return m_capacity;
    }

    bool empty() const
    {
        return m_size == 0;
    }

    void pop_back()
    {
        ASSERT(m_size > 0);
        m_size--;
        m_buffer[m_size] = 0;
    }

    T& operator[](const size_t idx)
    {
        ASSERT(idx < m_size);
        return m_buffer[idx];
    }

    const T& operator[](const size_t idx) const
    {
        ASSERT(idx < m_size);
        return m_buffer[idx];
    }

    void clear()
    {
        if (m_buffer != emptyBuffer())
            deallocate(m_buffer, m_capacity);
        makeEmpty();
    }

    T* data() const
    {
        return m_buffer;
    }

    void resizeWithUninitializedValues(size_t newSize)
    {
        if (newSize) {
            ensureCapacity(newSize);
            if (newSize > m_size) {
                // leave new values uninitialized as per contract
            }
            m_size = newSize;
            m_buffer[m_size] = 0;
        } else {
            clear();
        }
    }

    T* takeBuffer()
    {
        T* buf = m_buffer;
        size_t oldCapacity = m_capacity;
        makeEmpty();
        // Caller takes ownership of returned buffer which was allocated with Allocator.
        // When it is not dynamically allocated (emptyBuffer), return a duplicate minimal buffer.
        if (buf == emptyBuffer()) {
            T* dup = allocate(0);
            if (buf != emptyBuffer())
                deallocate(buf, oldCapacity);
            return dup;
        }
        return buf;
    }

protected:
    T* allocate(size_t siz) const
    {
        T* ret = Allocator().allocate(siz + 1);
        ret[siz] = 0;
        return ret;
    }

    void deallocate(T* buffer, size_t siz)
    {
        Allocator().deallocate(buffer, siz + 1);
    }

private:
    T* m_buffer;
    size_t m_size;
    size_t m_capacity;
};
} // namespace Escargot

#endif
